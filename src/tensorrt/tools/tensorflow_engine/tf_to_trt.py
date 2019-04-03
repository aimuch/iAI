#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import division
from __future__ import print_function
import os
import sys
from random import randint
import numpy as np
from tqdm import tqdm
import cv2

try:
    from PIL import Image
    import pycuda.driver as cuda
    import pycuda.autoinit
    import argparse
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({}) 
Please make sure you have pycuda and the example dependencies installed. 
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]
""".format(err))
    exit(1)

try:
    import uff
except ImportError:
    raise ImportError("""Please install the UFF Toolkit""")

try:
    import tensorrt as trt
    from tensorrt.parsers import uffparser
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({}) 
Please make sure you have the TensorRT Library installed 
and accessible in your LD_LIBRARY_PATH
""".format(err))
    exit(1)


# >>>>>> Here need to modify based on your data >>>>>>
TEST_PATH = "./data/test/"
ENGINE_PATH = "./model/engine/model.engine"
frozen_model_path = "./model/frozen_graphs/model.pb"
UFF_PATH = "./model.uff"

frozen_input_name = "input"
frozen_node_name = ["fc_3/frozen"]

NET_INPUT_SHAPE = (128, 128)
NET_INPUT_IMAGE_SHAPE = (3, 128, 128)  # (C, H, W)
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


def test_Loader(test_path, net_input_shape):
    label_list = []
    img_list = []
    result = []
    folders = os.listdir(test_path)
    for folder in folders:
        folder_path = os.path.join(test_path, folder)
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


# API CHANGE: Try to generalize into a utils function
#Run inference on device

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

def infer(context, input_img, batch_size):
    #load engine
    engine = context.get_engine()
    assert(engine.get_nb_bindings() == 2)
    
    #create output array to receive data
    dims = engine.get_binding_dimensions(1).to_DimsCHW()
    elt_count = dims.C() * dims.H() * dims.W() * batch_size
    
    #convert input data to Float32
    input_img = input_img.astype(np.float32)
    
    #Allocate pagelocked memory
    output = cuda.pagelocked_empty(elt_count, dtype=np.float32)

    #alocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.size * input_img.dtype.itemsize)
    d_output = cuda.mem_alloc(batch_size * output.size * output.dtype.itemsize)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    #transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    #execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    #transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)

    #return predictions
    return output

def main():
    MAX_WORKSPACE = 1 << 30
    MAX_BATCHSIZE = 1
    # 若用了output_filename参数则返回的是NULL，否则返回的是序列化以后的UFF模型数据
    uff_model = uff.from_tensorflow_frozen_model(frozen_model_path, frozen_node_name) #, output_filename=UFF_PATH, text=True, list_nodes=True)
    
    parser = uffparser.create_uff_parser()
    parser.register_input(frozen_input_name, NET_INPUT_IMAGE_SHAPE, 0) # 0表示输入通道顺序NCHW,1表示输入通道顺序为NHWC
    parser.register_output(frozen_node_name[0])
    
    engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, MAX_BATCHSIZE, MAX_WORKSPACE)

    # save engine
    trt.utils.write_engine_to_file(ENGINE_PATH, engine.serialize())

    assert(engine)

    # parser.destroy()
    context = engine.create_execution_context()

    print("\n| TEST CASE | PREDICTION |")
    pair = imgTestData[0]
    correct = 0
    for img,label in pair:
        output = infer(context, img, 1)

        # my frozen graph output is logists , here need convert to softmax
        softmax = np.exp(output) / np.sum(np.exp(output))
        predict = np.argmax(softmax)
        
        if int(label) == predict:
            correct += 1
        print("|-------|--------|--------------------------------------------------------")
        print("|   " + str(label) + "   |    " + str(predict) + "   |    " + str(['{:.2f}%'.format(i*100) for i in softmax]) + "   ")
    
    accuracy = correct/len(pair)
    print("Accuracy = ", accuracy)

if __name__ == "__main__":
    main()
