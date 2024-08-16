import logging
import threading
import subprocess

import numpy as np
import cv2

USB_GSTREAMER = True

def add_camera_args(parser):
    """Add parser augument for camera options."""
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=0, type=int)#默认的device为1,即外接usb摄像头
    parser.add_argument('--width', dest='image_width',
                        help='image width',
                        default=1920, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height',
                        default=1080, type=int)

    return parser




def open_cam_usb(dev, width, height):
    """Open a USB webcam."""
    if USB_GSTREAMER:
        gst_str = ("v4l2src device=/dev/video{} io-mode=2 ! image/jpeg, width={}, height={} ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink").format(dev, width, height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture(dev)


def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running:
        _, x = cam.cap.read()
        #x=cv2.flip(x,1)
        x=cv2.resize(x,(0,0),fx=0.5,fy=0.5)#因为原画显示及运算较大，乃NX cpu的小身板不能承受之重，帧率不理想，因此先缩放一下，不影响准确度。
        cam.img_handle=x
        
        
        if cam.img_handle is None:
            logging.warning('grab_img(): cap.read() returns None...')
            break
    cam.thread_running = False


class Camera():
    
    def __init__(self, args):
        self.args = args
        self.is_opened = False
        self.use_thread = False
        self.thread_running = False
        self.img_handle = None
        self.image_display = None
        self.img_width = 0
        self.img_height = 0
        self.cap = None
        self.thread = None
        #-----#
        self.vwriter = None

    def open(self):
        """Open camera based on command line arguments."""
        assert self.cap is None, 'Camera is already opened!'
        args = self.args

        self.cap = open_cam_usb(
            args.video_dev,
            args.image_width,
            args.image_height
        )
        self.use_thread = True
 
        if self.cap != 'OK':
            if self.cap.isOpened():
                # Try to grab the 1st image and determine width and height
                _, img = self.cap.read()
                if img is not None:
                    #img = frame=cv2.cvtColor(img,cv2.COLOR_YUV2BGR_I420)
                    self.img_height, self.img_width, _ = img.shape
                    self.is_opened = True

    #-------thread-----------------#
    def start(self):
        assert not self.thread_running
        if self.use_thread:
            self.thread_running = True
            self.thread = threading.Thread(target=grab_img, args=(self,))
            self.thread.start()

    def stop(self):
        self.thread_running = False
        if self.use_thread:
            self.thread.join()

    def read(self):
        #return cv2.resize(self.img_handle,(640,360))##!!
        return self.img_handle

    def write(self, frame):
        if self.vwriter:
            self.vwriter.write(frame)

    def release(self):
        # if self.cap != 'OK':
        self.cap.release()
        if self.vwriter is not None:
            self.vwriter.release()

