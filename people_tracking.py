import sys
PATH_LIST = ['/opt/ros/melodic/lib/python2.7/dist-packages', '/home/ysc/.local/lib/python3.6/site-packages']
for p in PATH_LIST:
    if p not in sys.path:
        sys.path.append(p)
import argparse
import cv2
import time
import rospy
import threading
import numpy as np
import depthai as dai
from yolo.tracking import Tracker
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from utils.parser import get_config

WINDOW_NAME = 'TrtYolo_deepsort'
develop = 1 # 1:开发者模式 关闭推流 打开窗口  #0: 部署模式 关闭窗口 打开推流

def parse_args():
    """Parse camera and input setting arguments."""
    desc = ("YOLO+DEEPSORT with TensorRT")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--engine', type=str, default='./weights/yolov8s.engine',
                        help='set your engine file path to load')
    parser.add_argument('--config_deepsort', type=str, default="./configs/deep_sort.yaml")
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')

    args = parser.parse_args()
    return args

def timeDeltaToS(delta) -> float:
    return delta.total_seconds()

class CvBridge():
    def __init__(self):
        self.numpy_type_to_cvtype = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
                                        'int16': '16S', 'int32': '32S', 'float32': '32F',
                                        'float64': '64F'}
        self.numpy_type_to_cvtype.update(dict((v, k) for (k, v) in self.numpy_type_to_cvtype.items()))

    def dtype_with_channels_to_cvtype2(self, dtype, n_channels):
        return '%sC%d' % (self.numpy_type_to_cvtype[dtype.name], n_channels)

    def cv2_to_imgmsg(self, cvim, encoding = "passthrough"):
        img_msg = Image()
        img_msg.height = cvim.shape[0]
        img_msg.width = cvim.shape[1]
        if len(cvim.shape) < 3:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, 1)
        else:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, cvim.shape[2])
        if encoding == "passthrough":
            img_msg.encoding = cv_type
        else:
            img_msg.encoding = encoding

        if cvim.dtype.byteorder == '>':
            img_msg.is_bigendian = True
        img_msg.data = cvim.tobytes()
        img_msg.step = len(img_msg.data) // img_msg.height
        return img_msg

FPS = 10

cam_list = {
    # "CAM_A": {"color": True, "res": "1200"},
    # "CAM_B": {"color": True, "res": "1200"},
    "CAM_C": {"color": True, "res": "1200"},
    "CAM_D": {"color": True, "res": "1200"},
}

mono_res_opts = {
    "400": dai.MonoCameraProperties.SensorResolution.THE_400_P,
    "480": dai.MonoCameraProperties.SensorResolution.THE_480_P,
    "720": dai.MonoCameraProperties.SensorResolution.THE_720_P,
    "800": dai.MonoCameraProperties.SensorResolution.THE_800_P,
    "1200": dai.MonoCameraProperties.SensorResolution.THE_1200_P,
}

color_res_opts = {
    "720": dai.ColorCameraProperties.SensorResolution.THE_720_P,
    "800": dai.ColorCameraProperties.SensorResolution.THE_800_P,
    "1080": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    "1200": dai.ColorCameraProperties.SensorResolution.THE_1200_P,
    "4k": dai.ColorCameraProperties.SensorResolution.THE_4_K,
    "5mp": dai.ColorCameraProperties.SensorResolution.THE_5_MP,
    "12mp": dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    "48mp": dai.ColorCameraProperties.SensorResolution.THE_48_MP,
}

cam_socket_to_name = {
    "RGB": "CAM_A",
    "LEFT": "CAM_B",
    "RIGHT": "CAM_C",
    "CAM_A": "CAM_A",
    "CAM_B": "CAM_B",
    "CAM_C": "CAM_C",
    "CAM_D": "CAM_D",
}

cam_socket_opts = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
    "CAM_D": dai.CameraBoardSocket.CAM_D,
}

pub_dict = {}
rospy.init_node('people_tracking', anonymous=True)
rate = rospy.Rate(2000)
for camera_name in ['CAM_C', 'CAM_D']:
    pub_dict[camera_name] = rospy.Publisher('/img/' + str(camera_name)+"/image_raw", Image, queue_size=0)

def ros_publish_img(img_ori, cam_name, timestamp, dt_time, bridge):
    if dt_time == -1:
        time_now = rospy.Time.now().to_sec()  # 当前ros时间
        cam_time = timeDeltaToS(timestamp)  # 当前相机时间
        dt_time = time_now - cam_time  # 计算出当前相机时间加上dt为当前ros时间（相机和IMU哪个先来就对齐哪个）
        cam_time = time_now
    else:
        cam_time = timeDeltaToS(timestamp)
        cam_time += dt_time  # 加速度时间加上dt变为ros时间

    img = bridge.cv2_to_imgmsg(img_ori, encoding="bgr8")
    img.header = Header()
    img.header.stamp.secs = int(cam_time)
    img.header.stamp.nsecs = int(
        (cam_time - int(cam_time)) * 1000 * 1000 * 1000
    )
    if dt_time != -1:
        pub_dict[cam_name].publish(img)

    return dt_time

def create_pipeline():
    pipeline = dai.Pipeline()
    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName('control')

    cam = {}
    xout = {}
    for cam_name, cam_props in cam_list.items():
        xout[cam_name] = pipeline.create(dai.node.XLinkOut)
        xout[cam_name].setStreamName(cam_name)
        if cam_props["color"]:
            cam[cam_name] = pipeline.create(dai.node.ColorCamera)
            cam[cam_name].setResolution(color_res_opts[cam_props["res"]])
            cam[cam_name].setIspScale(3,5) # 1080P -> 720P
            cam[cam_name].isp.link(xout[cam_name].input)
        else:
            cam[cam_name] = pipeline.createMonoCamera()
            cam[cam_name].setResolution(mono_res_opts[cam_props["res"]])
            cam[cam_name].out.link(xout[cam_name].input)
        cam[cam_name].setBoardSocket(cam_socket_opts[cam_name])
        cam[cam_name].setFps(FPS)
        # cam[cam_name].initialControl.setExternalTrigger(4, 3)
        cam[cam_name].initialControl.setFrameSyncMode(
                dai.CameraControl.FrameSyncMode.INPUT,
            )

    script = pipeline.create(dai.node.Script)
    script.setProcessor(dai.ProcessorType.LEON_CSS)
    script.setScript(
        """# coding=utf-8
import time
import GPIO

# Script static arguments
fps = %f

calib = Device.readCalibration2().getEepromData()
prodName  = calib.productName
boardName = calib.boardName
boardRev  = calib.boardRev

node.warn(f'Product name  : {prodName}') 
node.warn(f'Board name    : {boardName}') 
node.warn(f'Board revision: {boardRev}')

revision = -1
# Very basic parsing here, TODO improve
if len(boardRev) >= 2 and boardRev[0] == 'R':
    revision = int(boardRev[1])
node.warn(f'Parsed revision number: {revision}')

# Defaults for OAK-FFC-4P older revisions (<= R5)
GPIO_FSIN_2LANE = 41  # COM_AUX_IO2
GPIO_FSIN_4LANE = 40
GPIO_FSIN_MODE_SELECT = 6  # Drive 1 to tie together FSIN_2LANE and FSIN_4LANE

if revision >= 6:
    GPIO_FSIN_2LANE = 41  # still COM_AUX_IO2, no PWM capable
    GPIO_FSIN_4LANE = 42  # also not PWM capable
    GPIO_FSIN_MODE_SELECT = 38  # Drive 1 to tie together FSIN_2LANE and FSIN_4LANE
# Note: on R7 GPIO_FSIN_MODE_SELECT is pulled up, driving high isn't necessary (but fine to do)

# GPIO initialization
GPIO.setup(GPIO_FSIN_2LANE, GPIO.OUT)
GPIO.write(GPIO_FSIN_2LANE, 0)

GPIO.setup(GPIO_FSIN_4LANE, GPIO.IN)

GPIO.setup(GPIO_FSIN_MODE_SELECT, GPIO.OUT)
GPIO.write(GPIO_FSIN_MODE_SELECT, 1)

period = 1 / fps
active = 0.001

node.warn(f'FPS: {fps}  Period: {period}')

withInterrupts = False
if withInterrupts:
    node.critical(f'[TODO] FSYNC with timer interrupts (more precise) not implemented')
else:
    overhead = 0.003  # Empirical, TODO add thread priority option!
    while True:
        GPIO.write(GPIO_FSIN_2LANE, 1)
        time.sleep(active)
        GPIO.write(GPIO_FSIN_2LANE, 0)
        time.sleep(period - active - overhead)
""" % (FPS)
    )

    return pipeline

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

def loop_and_track(device, tracker, bridge):
    global cam_list
    strID = ""

    # 创建相机输出队列
    output_queues = {}
    for cam_name in cam_list:
        output_queues[cam_name] = device.getOutputQueue(
            name=cam_name, maxSize=4, blocking=False,
        )
    CAM_LEFT = "CAM_D"
    CAM_RIGHT = "CAM_C"

    # 读取矫正参数
    rectify_map_file = "./rectify_map.npz"
    rectify_map = np.load(rectify_map_file)
    mapx_L = rectify_map['mapx_L']
    mapy_L = rectify_map['mapy_L']
    mapx_R = rectify_map['mapx_R']
    mapy_R = rectify_map['mapy_R']

    # 循环读取视频流
    dt_time = -1
    while (not device.isClosed()) and (not rospy.is_shutdown()):
        start = time.time()

        # left image
        packet = output_queues[CAM_LEFT].tryGet()
        if packet is not None:
            # 输出视频帧的时间戳
            timestamp = packet.getTimestampDevice()
            img_cv = packet.getCvFrame()
            left_img = cv2.flip(img_cv, 0)
            dt_time = ros_publish_img(left_img, CAM_LEFT, timestamp, dt_time, bridge)

        # right image
        packet = output_queues[CAM_RIGHT].tryGet()
        if packet is not None:
            # 输出视频帧的时间戳
            timestamp = packet.getTimestampDevice()
            img_cv = packet.getCvFrame()
            right_img = cv2.flip(img_cv, 0)
            dt_time = ros_publish_img(right_img, CAM_RIGHT, timestamp, dt_time, bridge)

        # 极线矫正
        left_img_rect = cv2.remap(left_img, mapx_L, mapy_L, cv2.INTER_CUBIC)
        right_img_rect = cv2.remap(right_img, mapx_R, mapy_R, cv2.INTER_CUBIC)
        
        # 检测跟踪
        img_final = tracker.run(left_img_rect, develop)

        # 可视化
        end = time.time()
        img_final = cv2.putText(
            img_final, "fps {:.02f}".format(1 / (end - start)),
            (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, [0, 128, 0], 2
        )
        cv2.imshow(WINDOW_NAME, img_final)
        key = cv2.waitKey(10)
        
        # print(key) ?
        if key == -1:
            continue
        elif key == 27:  # ESC key: quit program
            break
        elif key >= ord('0') and key <= ord('9'):  # 大键盘输入
            strID += str((int(key - ord('0'))))
        elif key >= 176 and key <= 185:  # 小键盘输入
            strID += str((int(key - 176)))
        elif key == ord('\b'):  # 退格
            strID = strID[:-1]
        elif key == 10 or key == 13 or key == 141:  # 回车
            if strID == '':
                tracker.Id = 0
                continue
            tracker.Id = int(strID)
            print(int(strID))
            strID = ""

def spin_thread():
    rospy.spin()

def main():
    args = parse_args()
    det_cfg = args
    track_cfg = get_config()
    track_cfg.merge_from_file(args.config_deepsort)

    global cam_list
    bridge = CvBridge()

    with dai.Device() as device:
        print(device.getIrDrivers())
        print(device.setIrFloodLightBrightness(0))
        print(device.setIrLaserDotProjectorBrightness(1000))
        # 获取连接到设备上的相机列表，输出相机名称、分辨率、支持的颜色类型等信息
        print("Connected cameras:")
        sensor_names = {}  # type: dict[str, str]
        for p in device.getConnectedCameraFeatures():
            # 输出相机信息
            print(
                f" -socket {p.socket.name:6}: {p.sensorName:6} {p.width:4} x {p.height:4} focus:",
                end="",
            )
            print("auto " if p.hasAutofocus else "fixed", "- ", end="")
            supported_types = [color_type.name for color_type in p.supportedTypes]
            print(*supported_types)

            # 更新相机属性表
            cam_name = cam_socket_to_name[p.socket.name]
            sensor_names[cam_name] = p.sensorName

        # 仅保留设备已连接的相机
        for cam_name in set(cam_list).difference(sensor_names):
            print(f"{cam_name} is not connected !")

        cam_list = {
            name: cam_list[name] for name in set(cam_list).intersection(sensor_names)
        }

        # 开始执行给定的管道
        pipeline = create_pipeline()
        device.startPipeline(pipeline)

        controlQueue = device.getInputQueue('control')
        
        # Defaults and limits for manual focus/exposure controls
        expTime = 1000  # us
        expMin = 1
        expMax = 33000

        sensIso = 250
        sensMin = 100
        sensMax = 1600

        expStep = 500  # us
        isoStep = 50

        wbManual = 3500

        expTime = clamp(expTime, expMin, expMax)
        sensIso = clamp(sensIso, sensMin, sensMax)
        print("Setting manual exposure, time:", expTime, "iso:", sensIso)

        ctrl = dai.CameraControl()
        ctrl.setManualExposure(expTime, sensIso)
        ctrl.setManualWhiteBalance(wbManual)
        controlQueue.send(ctrl)

        add_thread = threading.Thread(target=spin_thread)
        add_thread.start()

        tracker = Tracker(det_cfg, track_cfg)

        print('start loop track')
        loop_and_track(device, tracker, bridge)

        print('close')


if __name__ == '__main__':
    main()
