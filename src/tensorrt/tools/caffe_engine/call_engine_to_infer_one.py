import os
# import tensorflow as tf
import tensorrt as trt
from tensorrt.parsers import uffparser
import pycuda.driver as cuda
# import uff
import cv2
from PIL import Image
import numpy as np

IMG_PATH = "/media/andy/Data/DevWorkSpace/Projects/imageClassifier/data/test/valid/parallel_1540_0_877328340.png"
LABEL = 0
ENGINE_PATH = "/home/andy/caffe/examples/mydata/slot_classifier/engine/px2_classifier.engine"
NET_INPUT_SHAPE = (256, 256)
NET_OUTPUT_SHAPE = 5


# Load Image
def load_image(img_path, net_input_shape):
    img = cv2.resize(cv2.imread(img_path), net_input_shape)
    # BGR -> RGB
    #img = img[:,:, (2, 1, 0)]

    ## Method 1
    # imgT = np.transpose(img, (2, 0, 1))  # c,w,h
    # imgF = np.asarray(imgT, dtype=np.float32)
    # mean = [[[88.159309]], [[97.966286]], [[103.66106]]] # Caffe image mean
    # imgS = np.subtract(imgF,mean)

    ## Method 2
    imgF = np.asarray(img, dtype=np.float32)
    mean = [88.159309, 97.966286, 103.66106] # Caffe image mean
    imgSS = np.subtract(imgF, mean)
    imgS = np.transpose(imgSS, (2, 0, 1))  # CHW

    # RGB_MEAN_PIXELS = np.array([88.159309, 97.966286, 103.66106]).reshape((1,1,1,3)).astype(np.float32)

    return np.ascontiguousarray(imgS, dtype=np.float32)   # avoid error: ndarray is not contiguous


img = load_image(IMG_PATH, NET_INPUT_SHAPE)

# Load Engine file
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
engine = trt.utils.load_engine(G_LOGGER, ENGINE_PATH)
context = engine.create_execution_context()
runtime = trt.infer.create_infer_runtime(G_LOGGER)

output = np.empty(NET_OUTPUT_SHAPE, dtype = np.float32)

# Alocate device memory
d_input = cuda.mem_alloc(1 * img.nbytes) # img.size*img.dtype.itemsize==img.nbytes
d_output = cuda.mem_alloc(1 * output.nbytes) # output.size*output*itemsize = output.nbytes

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

print("softmax = ", output)
# print("Test Case: " + str(LABEL))
print ("Prediction: " + str(np.argmax(output)))