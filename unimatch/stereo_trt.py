import time
import cv2
import numpy as np
import tensorrt as trt

import sys
sys.path.append('//usr/src/tensorrt/samples/python')
import common


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


class Depth_Estimator():
    def __init__(self, engine_path, input_height, input_width):
        self.engine = self.get_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.input_height = input_height
        self.input_width = input_width
        self.K = np.array([
            [671.178344354735, 0, 569.657570856393],
            [0, 672.873159716253, 372.135461408947],
            [0, 0, 1]
        ])  # camera intrinsics
        self.baseline = 99.1863877037785 * 1e-3  # baseline length in meters

    @staticmethod
    def get_engine(engine_file_path):
        print(f"\033[32mReading engine from file {engine_file_path}\033[0m")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def compute_disparity(self, left_image, right_image, return_color=False):
        ori_height = left_image.shape[0]
        ori_width = left_image.shape[1]

        left = cv2.resize(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height)).astype(np.float32) / 255.0
        right = cv2.resize(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height)).astype(np.float32) / 255.0

        left = np.transpose(left, (2, 0, 1))[np.newaxis, :, :, :]
        right = np.transpose(right, (2, 0, 1))[np.newaxis, :, :, :]
        
        self.inputs[0].host = np.ascontiguousarray(left)
        self.inputs[1].host = np.ascontiguousarray(right)
        outputs = common.do_inference_v2(
            self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream
        )
        
        disp = outputs[0].reshape(self.input_height, self.input_width)
        disp = cv2.resize(disp, (ori_width, ori_height))
        
        if not return_color:
            return disp, None
        else:
            norm = ((disp - disp.min()) / (disp.max() - disp.min()) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
            return disp, colored
    
    def generate_target_point(self, disp, target_bbox):
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        
        x1, y1, x2, y2 = target_bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

        disp = disp[y1:y2, x1:x2]
        depth = (self.baseline * fx) / disp

        # apply Otsu's thresholding to segment the foreground object
        normal_depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        _, otsu_thresh = cv2.threshold(normal_depth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground_mask = (otsu_thresh == 0)
        # cv2.imshow('foreground_mask', foreground_mask.astype(np.uint8) * 255)
        # cv2.waitKey(1)

        # calculate the depth of the foreground object
        foreground_depth = depth[foreground_mask]
        depth = np.mean(foreground_depth)

        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        BEV_X = Z
        BEV_Y = -X
        target_point = (BEV_X, BEV_Y)

        return target_point

