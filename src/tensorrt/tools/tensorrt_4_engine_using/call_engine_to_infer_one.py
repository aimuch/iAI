#!/usr/bin/python
# -*- coding: UTF-8 -*-

# TensorRT Version <= 4

import os
# import tensorflow as tf
import tensorrt as trt
from tensorrt.parsers import uffparser
import pycuda.driver as cuda
# import uff
import cv2
import numpy as np
from tqdm import tqdm


# >>>>>> Here need to modify based on your data >>>>>>
img_path = "/media/andy/Data/DevWorkSpace/Projects/imageClassifier/data/test/valid/parallel_2862_1_16547177.png"
LABEL = 0

ENGINE_PATH = "./model/engine/model.engine"
NET_INPUT_SHAPE = (128, 128)
NET_OUTPUT_SHAPE = 5
class_labels = ['error', 'half', 'invlb', 'invls', 'valid']
# <<<<<< Here need to modify based on your data <<<<<<


# Load Image
def load_image(img_path, net_input_shape):
    # Use the same pre-processing as training
    img = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), NET_INPUT_SHAPE)
    img = (img-128.)/128.

    # Fixed usage
    img = np.transpose(img, (2, 0, 1)) # 要转换成CHW,这里要特别注意
    return np.ascontiguousarray(img, dtype=np.float32) # 避免error:ndarray is not contiguous


img = load_image(img_path, NET_INPUT_SHAPE)
print(img_path)
# Load Engine file
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
engine = trt.utils.load_engine(G_LOGGER, ENGINE_PATH)
context = engine.create_execution_context()
runtime = trt.infer.create_infer_runtime(G_LOGGER)


output = np.empty(NET_OUTPUT_SHAPE, dtype = np.float32)

# Alocate device memory
d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize) # img.size * img.dtype.itemsize=img.nbytes
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)  # output.size * output.dtype.itemsize=output.nbytes

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

# Transfer input data to device
cuda.memcpy_htod_async(d_input, img, stream)

# Execute model 
context.enqueue(1, bindings, stream.handle, None)

# Transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)
# Syncronize threads
stream.synchronize()


# my frozen graph output is logists , here need convert to softmax
softmax = np.exp(output) / np.sum(np.exp(output))
predict = np.argmax(softmax)

print("True = ",LABEL, ", predict = ", predict, ", softmax = ", softmax)
