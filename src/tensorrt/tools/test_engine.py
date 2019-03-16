import os
# import tensorflow as tf
import tensorrt as trt
from tensorrt.parsers import uffparser
import pycuda.driver as cuda
# import uff
from PIL import Image
import numpy as np



IMG_PATH = "/media/andy/Data/DevWorkSpace/Projects/imageClassifier/data/test/error/parallel_797_0_95235061.png"
LABEL = 0
ENGINE_PATH = "/media/andy/Data/DevWorkSpace/Projects/imageClassifier/model/uff/model.engine"
NET_INPUT_SHAPE = (32, 32)
NET_OUTPUT_SHAPE = 5


def normalize_img(img):
    """
    Normalize image data to [-1,+1]
    
    Arguments:
        img: source image
    """
    return (img-128.)/128.

# Load Image
def load_image(img_path, net_input_shape):
    img = Image.open(img_path)
    img = img.resize(net_input_shape)
    return np.asarray(img, dtype=np.float32)


img = load_image(IMG_PATH, NET_INPUT_SHAPE)
img = normalize_img(img)

# Load Engine file
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
engine = trt.utils.load_engine(G_LOGGER, ENGINE_PATH)
context = engine.create_execution_context()
runtime = trt.infer.create_infer_runtime(G_LOGGER)

output = np.empty(5, dtype = np.float32)

# Alocate device memory
d_input = cuda.mem_alloc(1 * img.nbytes)
d_output = cuda.mem_alloc(NET_OUTPUT_SHAPE * output.nbytes)

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


print("Test Case: " + str(LABEL))
print ("Prediction: " + str(np.argmax(output)))
