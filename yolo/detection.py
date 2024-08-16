import argparse
import cv2
import torch
import rospy
import time

from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from config import CLASSES_DET, COLORS
from models import TRTModule  # isort:skip
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list

WINDOW_NAME = 'TrtYolo_detection'

def image_callback(data, context):
    Engine, H, W, device, show= context['engine'], context['H'], context['W'], context['device'], context['show']
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    img = cv2.flip(cv_image, 0)
    draw = img.copy()

    # inference
    start = time.time()
    img, ratio, dwdh = letterbox(img, (W, H))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
    tensor = torch.asarray(tensor, device=device)
    data = Engine(tensor)
    end = time.time()
        
    bboxes, scores, labels = det_postprocess(data)
    if bboxes.numel() == 0:
        # if no bounding box
        print('no object!')
    else:
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            if int(label) == 0:
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                cls = CLASSES_DET[cls_id]
                color = COLORS[cls]

                text = f'{cls}:{score:.3f}'
                x1, y1, x2, y2 = bbox

                (_w, _h), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
                _y1 = min(y1 + 1, draw.shape[0])

                draw = cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                draw = cv2.rectangle(draw, (x1, _y1), (x1 + _w, _y1 + _h + _bl), (0, 0, 255), -1)
                draw = cv2.putText(draw, text, (x1, _y1 + _h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                draw = cv2.putText(draw, "fps {:.02f}".format(1 / (end - start)), (10, 20),
                                        cv2.FONT_HERSHEY_PLAIN, 2, [0, 128, 0], 2)

    img_draw = bridge.cv2_to_imgmsg(draw, encoding="bgr8")
    image_pub.publish(img_draw)
    if show:
        cv2.imshow(WINDOW_NAME, draw)
        cv2.waitKey(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]
    show = args.show
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
    rospy.init_node('people_tracking', anonymous=True)
    image_sub = rospy.Subscriber("/img/CAM_C/image_raw", Image, image_callback, callback_args={'engine': Engine, 'H': H, 'W': W, 'device': device, 'show': show})
    image_pub = rospy.Publisher("/detection_result", Image, queue_size=0)
    pub_box = rospy.Publisher('/box', Int32MultiArray, queue_size=10)
    rospy.spin()
    
    