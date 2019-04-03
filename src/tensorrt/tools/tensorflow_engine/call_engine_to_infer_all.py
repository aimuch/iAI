#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
# import tensorflow as tf
import tensorrt as trt
import uff
from tensorrt.parsers import uffparser
import pycuda.driver as cuda
# import uff
import cv2
import numpy as np
from tqdm import tqdm


# >>>>>> Here need to modify based on your data >>>>>>
TEST_PATH = "./data/test/"
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

def test_Loader(TEST_PATH, net_input_shape):
    label_list = []
    img_list = []
    result = []
    folders = os.listdir(TEST_PATH)
    for folder in folders:
        folder_path = os.path.join(TEST_PATH, folder)
        imgs = os.listdir(folder_path)
        for img in tqdm(imgs):
            img_path = os.path.join(folder_path, img)
            img = load_image(img_path, net_input_shape)
            label = class_labels.index(folder)
            img_list.append(img)
            label_list.append(label)
            result.append((img, label))
    
    return result, (img_list, label_list)


imgTestData = test_Loader(TEST_PATH, NET_INPUT_SHAPE)

# Load Engine file
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)


# >>>> uff -> engine >>>>
# frozen_model_path = "./model/frozen_graphs/model.pb"
# uff_model = uff.from_tensorflow_frozen_model(frozen_model_path,["fc3/frozen"])
# parser = uffparser.create_uff_parser()
# parser.register_input("input", (3, 128, 128), 0) # 0表示输入通道顺序NCHW,1表示输入通道顺序为NHWC
# parser.register_output("fc3/frozen")
# engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1<<30)
# <<<< uff -> engine <<<<


# >>>> load engine >>>>
engine = trt.utils.load_engine(G_LOGGER, ENGINE_PATH)
# <<<< load engine <<<<


runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

predicts = []
pair = imgTestData[0]
for img, label in pair:
    output = np.empty(NET_OUTPUT_SHAPE, dtype = np.float32)

    # Alocate device memory
    d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize) # img.size * img.dtype.itemsize == img.nbytes
    d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize) # output.size * output.dtype.itemsize == output.nbytes

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
    predicts.append(predict)

    print("|-------|--------|--------------------------------------------------------")
    print("|   " + str(label) + "   |    " + str(predict) + "   |    " + str(['{:.2f}%'.format(i*100) for i in softmax]) + "   ")


grandTrue = np.array(imgTestData[1][1])
predicts = np.array(predicts)
error = predicts[predicts!=grandTrue]


print("Sample = ", len(pair), "error = ", len(error))
print("Accuracy = ",(len(pair)-len(error))/len(pair))


context.destroy()
engine.destroy()
runtime.destroy()