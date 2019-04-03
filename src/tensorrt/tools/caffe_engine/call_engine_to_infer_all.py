import os
# import tensorflow as tf
import tensorrt as trt
from tensorrt.parsers import uffparser
import pycuda.driver as cuda
# import uff
import cv2
import numpy as np
from tqdm import tqdm



TEST_PATH = "/media/andy/Data/DevWorkSpace/Projects/imageClassifier/data/test/"
LABEL = 0
ENGINE_PATH = "/home/andy/caffe/examples/mydata/slot_classifier/engine/px2_classifier.engine"
NET_INPUT_SHAPE = (256, 256)
NET_OUTPUT_SHAPE = 5
class_labels = ['error', 'half', 'invlb', 'invls', 'valid']

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

    return np.ascontiguousarray(imgS, dtype=np.float32) # avoid error: ndarray is not contiguous

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
engine = trt.utils.load_engine(G_LOGGER, ENGINE_PATH)
context = engine.create_execution_context()
runtime = trt.infer.create_infer_runtime(G_LOGGER)

# output = np.empty(1, dtype = np.float32)

# # Alocate device memory
# d_input = cuda.mem_alloc(1 * imgTestData[0][0][0].nbytes)
# d_output = cuda.mem_alloc(NET_OUTPUT_SHAPE * output.nbytes)

# bindings = [int(d_input), int(d_output)]

# stream = cuda.Stream()

predicts = []
pair = imgTestData[0]
for img, label in pair:
    output = np.empty(NET_OUTPUT_SHAPE, dtype = np.float32)

    # Alocate device memory
    d_input = cuda.mem_alloc(1 * img.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

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

    softmax = np.exp(output) / np.sum(np.exp(output))
    predict = np.argmax(softmax)
    predicts.append(predict)

    print("True = ",label, ", predict = ", predict, ", softmax = ", softmax)


grandTrue = np.array(imgTestData[1][1])
predicts = np.array(predicts)
error = predicts[predicts!=grandTrue]

print(imgTestData[1][1])
print("-------")
print(predicts)
print("-------")
print(len(error))
print((len(imgTestData[0])-len(error))/len(imgTestData[0]))