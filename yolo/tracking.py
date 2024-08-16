import time
import rospy
import torch
import cv2
import random
import tensorrt as trt
import numpy as np

from std_msgs.msg import Int32MultiArray
from utils import common
from utils.data_processing import *
from utils.draw import draw_boxes
from deep_sort import build_tracker
from yolo.models import TRTModule  # isort:skip
from yolo.models.torch_utils import det_postprocess
from yolo.models.utils import blob, letterbox, path_to_list

TRT_LOGGER = trt.Logger()

pub_box = rospy.Publisher('/box', Int32MultiArray, queue_size=10)

def box_publisher(bbox_xyxy, identities, Id):
    msg = Int32MultiArray()
    array = [-1,0]
    for i,box in enumerate(bbox_xyxy):
        x1,y1,x2,y2 = [int(j) for j in box]
        array_temp = [x1,y1,x2,y2]
        array.extend(array_temp)
        id_temp = int(identities[i]) if identities is not None else 0
        array[1] = array[1] + 1
        if id_temp == Id:
            array[0] = i
    msg.data = array    
    pub_box.publish(msg)
        
class History():
    def __init__(self):
        self.lst=[]
        self.capnum=2
    def add(self,x):
        if x is None:
            a.pop(0)
            return
        if len(self.lst)<self.capnum:
            #print(x)
            self.lst.insert(-1,x) 
        
        else:
            self.lst[:-1]=self.lst[1:]
            self.lst[-1]=x 
            
    def predictx(self):
        if len(self.lst)<self.capnum//2:
            return None

        
        return self.lst[-1]
        
    def reset(self):
        del self.lst[:]


class Tracker():
    def __init__(self, det_cfg, track_cfg):
        self.Id = 0
        self.scoremax = 0

        print(det_cfg)
        self.deepsort = build_tracker(track_cfg, use_cuda=True)
        #---tensorrt----#
        self.device = torch.device(det_cfg.device)
        self.Engine = TRTModule(det_cfg.engine, self.device)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]
        self.show = det_cfg.show
        self.Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        # ---tensorrt----#

        self.his=History()
        print("Init_done")


    def run(self, ori_im, develop):
        #print("____"+str(self.Id)+"____")
        img = ori_im.copy()
        # inference
        start = time.time()
        ori_im, ratio, dwdh = letterbox(ori_im, (self.W, self.H))
        # print(ratio,dwdh)
        rgb = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=self.device)
        tensor = torch.asarray(tensor, device=self.device)
        data = self.Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        end = time.time()

        if bboxes is not None:#yolo has detected object
            # select person class
            mask = labels.int() == 0
            bboxes = bboxes[mask]
            #bbox_xywh[:, 3:] *= 1.2
            scores = scores[mask]
            if len(bboxes) > 0:#yolo has detected person
                bboxes -= dwdh
                bboxes /= ratio
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clamp(0, img.shape[1]-1)
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clamp(0, img.shape[0]-1)
                # do tracking
                outputs = self.deepsort.update(bboxes.cpu().numpy(), scores.cpu().numpy(), img)

                # draw boxes for visualization
                if len(outputs) > 0:#deepsort has figure out the person
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    # publish box topic              
                    box_publisher(bbox_xyxy, identities, self.Id)
                    
                    #ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                    ori_im = draw_boxes(img, bbox_xyxy, identities, self.Id,self.his, develop)
                else:
                    ori_im = draw_boxes(img, [], None, self.Id,self.his, develop)
            else:#no object
                #elif self.Id !=0:
                ori_im = draw_boxes(img, [], None, self.Id,self.his, develop)
        else:#no object
            #elif self.Id !=0:
            ori_im = draw_boxes(img, [], None, self.Id,self.his, develop)

        return ori_im
