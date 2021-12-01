#!/usr/bin/python
# -*- coding: UTF-8 -*-

# TensorRT Version 7.1.3

import tensorrt as trt
import pycuda.driver as cuda
#import pycuda.driver as cuda2
import pycuda.autoinit
import numpy as np
import cv2
import scipy.special

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
                        116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                        168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                        220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                        272, 276, 280, 284
                        ]   # 56

cls_num_per_lane = 56

def load_engine(engine_path):
    #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

path ='./ldw_fp32.engine'

# 1. 建立模型，构建上下文管理器
engine = load_engine(path)
context = engine.create_execution_context()
context.active_optimization_profile = 0

#2. 读取数据，数据处理为可以和网络结构输入对应起来的的shape，数据可增加预处理
imgpath = './img_to_test/7.jpg'
vis = cv2.imread(imgpath)
img_w, img_h = 1280, 720
image = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
image = cv2.resize(vis, (800, 288))         # => 800x288
image = image/255.0                         # => [0.0, 1.0]
image[:,:,0] = (image[:,:,0]-0.485) / 0.229
image[:,:,1] = (image[:,:,1]-0.456) / 0.224
image[:,:,2] = (image[:,:,2]-0.406) / 0.225
image = np.transpose(image, (2, 0, 1))      # HWC->CHW
# cv2.imwrite("./test1.jpg", image)
image = np.expand_dims(image, 0)            # Add batch dimension. NCHW
image = np.ascontiguousarray(image, dtype=np.float32)


#3.分配内存空间，并进行数据cpu到gpu的拷贝
#动态尺寸，每次都要set一下模型输入的shape，0代表的就是输入，输出根据具体的网络结构而定，可以是0,1,2,3...其中的某个头。
context.set_binding_shape(0, image.shape)
d_input = cuda.mem_alloc(image.nbytes)  #分配输入的内存。


output_shape = context.get_binding_shape(1)
buffer = np.empty(output_shape, dtype=np.float32)
d_output = cuda.mem_alloc(buffer.nbytes)    #分配输出内存。
cuda.memcpy_htod(d_input, image)
bindings = [d_input ,d_output]

#4.进行推理，并将结果从gpu拷贝到cpu。
context.execute_v2(bindings)  #可异步和同步
cuda.memcpy_dtoh(buffer,d_output)
output = buffer.reshape(output_shape)  # (num_grad + 1) * sample_rows * num_lane

#5.对推理结果进行后处理。这里只是举了一个简单例子，可以结合官方静态的yolov3案例完善。
col_sample = np.linspace(0, 800 - 1, 100)   # 横向格子个数
col_sample_w = col_sample[1] - col_sample[0] # 每个格子宽度

out_j = output[0][:, ::-1, :]                           #! reverse sample_rows dim
prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)  # out_j[:-1, :, :] only process num_grad dim
idx = np.arange(100) + 1
idx = idx.reshape(-1, 1, 1)
loc = np.sum(prob * idx, axis=0)
out_j = np.argmax(out_j, axis=0)
loc[out_j == 100] = 0
out_j = loc                                             # (sample_rows, num_lane)

# import pdb; pdb.set_trace()
for i in range(out_j.shape[1]):                         # num_lanes
    if np.sum(out_j[:, i] != 0) > 2:
        for k in range(out_j.shape[0]):                 # sample_rows
            if out_j[k, i] > 0:
                ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (tusimple_row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                cv2.circle(vis, ppp, 5, (0,255,0), -1)
cv2.imwrite("./result.jpg", vis)

print("Done")
