import tensorrt as trt
from utils import common
from utils.data_processing import *
from utils.draw import draw_boxes
from deep_sort import build_tracker
import numpy as np
import time
import rospy
from std_msgs.msg import Int32MultiArray

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

def get_engine(engine_file_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
        
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
    def __init__(self, args, engine_file_path):
        self.args = args
        self.Id = 0
        self.inputmethod = 1
        self.scoremax = 0
        # self.args = args

        self.deepsort = build_tracker(args, use_cuda=True)
        #---tensorrt----#
        self.device = torch.device(args.device)
        self.Engine = TRTModule(args.engine, device)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]
        self.show = args.show
        self.Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        # ---tensorrt----#

        self.his=History()


    def run(self, ori_im, develop):
        #print("____"+str(self.Id)+"____")

        image_raw, image = self.preprocessor.process(ori_im)
        shape_orig_WH = image_raw.size
        # print('type of image:',  type(image))

        self.inputs[0].host = image
        trt_outputs = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)

        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]
        bbox_xywh, cls_ids, cls_conf = self.postprocessor.process(trt_outputs, (shape_orig_WH))

        if self.targetflag == True:    
            if self.targetx < 0 or self.targety < 0 or self.targetx > ori_im.shape[1] or self.targety > ori_im.shape[0]:  #手柄发送重置指令，通信节点发送-1从而重置
                self.Id = 0

        if bbox_xywh is not None:#yolo has detected person
            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            #bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]
            
            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_im)

            # publish box topic

            # draw boxes for visualization
            if len(outputs) > 0:#deepsort has figure out the person
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                                
                box_publisher(bbox_xyxy, identities, self.Id)
                
                #ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities, self.Id,self.his, develop)
            else:
                ori_im = draw_boxes(ori_im, [], None, self.Id,self.his, develop)
        else:#no person
            #elif self.Id !=0:
            ori_im = draw_boxes(ori_im, [], None, self.Id,self.his, develop)

        return ori_im
