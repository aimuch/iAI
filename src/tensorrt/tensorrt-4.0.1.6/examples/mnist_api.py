#
# Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

#!/usr/bin/python
from random import randint
import numpy as np

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))

try:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    import argparse
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have pycuda and the example dependencies installed.
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]""".format(err))

try:
    import tensorrt as trt
    from tensorrt.parsers import caffeparser
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH""".format(err))

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
INPUT_LAYERS = ["data"]
OUTPUT_LAYERS = ['prob']
INPUT_H = 28
INPUT_W =  28
OUTPUT_SIZE = 10

PARSER = argparse.ArgumentParser(description="Example of how to create a Caffe based TensorRT Engine and run inference")
PARSER.add_argument('datadir', help='Path to Python TensorRT data directory (realpath)')

DATA_DIR = PARSER.parse_args().datadir

WEIGHTS_PATH = DATA_DIR + '/mnist/mnistapi.wts'
MEAN = DATA_DIR + '/mnist/mnist_mean.binaryproto'
DATA = DATA_DIR + '/mnist/'

def create_MNIST_engine(max_batch_size, builder, dt, weights_file):
    network = builder.create_network()

    data = network.add_input(INPUT_LAYERS[0], dt, (1, INPUT_H, INPUT_W))
    assert(data)

    scale_param = 0.0125

    scale_weight = np.empty(1, np.float32)
    scale_weight[0] = scale_param

    power = trt.infer.Weights.empty(trt.infer.DataType.FLOAT)
    shift = trt.infer.Weights.empty(trt.infer.DataType.FLOAT)
    scale = trt.infer.Weights(scale_weight)

    scale1 = network.add_scale(data, trt.infer.ScaleMode.UNIFORM, shift, scale, power)
    assert(scale1)

    weight_map = trt.utils.load_weights(weights_file)

    conv1 = network.add_convolution(scale1.get_output(0), 20, (5,5), weight_map["conv1filter"], weight_map["conv1bias"])
    assert(conv1)
    conv1.set_stride((1,1))

    pool1 = network.add_pooling(conv1.get_output(0), trt.infer.PoolingType.MAX, (2,2))
    assert(pool1)
    pool1.set_stride((2,2))

    conv2 = network.add_convolution(pool1.get_output(0), 50, (5,5), weight_map["conv2filter"], weight_map["conv2bias"])
    assert(conv2)
    conv2.set_stride((1,1))

    pool2 = network.add_pooling(conv2.get_output(0), trt.infer.PoolingType.MAX, (2,2))
    assert(pool2)
    pool2.set_stride((2,2))

    ip1 = network.add_fully_connected(pool2.get_output(0), 500, weight_map["ip1filter"], weight_map["ip1bias"])
    assert(ip1)

    relu1 = network.add_activation(ip1.get_output(0), trt.infer.ActivationType.RELU)
    assert(relu1)

    ip2 = network.add_fully_connected(relu1.get_output(0), OUTPUT_SIZE, weight_map["ip2filter"], weight_map["ip2bias"])
    assert(ip2)

    prob = network.add_softmax(ip2.get_output(0))
    assert(prob)
    prob.get_output(0).set_name(OUTPUT_LAYERS[0])
    network.mark_output(prob.get_output(0))

    builder.set_max_batch_size(max_batch_size)
    builder.set_max_workspace_size(1 << 20)
    # Uncomment for FP16 mode.
    # builder.set_fp16_mode(True)

    engine = builder.build_cuda_engine(network)
    network.destroy()
    return engine

def API_to_model(weights_file, max_batch_size):
    builder = trt.infer.create_infer_builder(G_LOGGER)
    engine = create_MNIST_engine(max_batch_size, builder, trt.infer.DataType.FLOAT, weights_file)
    assert(engine)

    modelstream = engine.serialize()
    engine.destroy()
    builder.destroy()
    return modelstream

# Run inference on device
def infer(context, input_img, output_size, batch_size):
    # Load engine
    engine = context.get_engine()
    assert(engine.get_nb_bindings() == 2)
    # Convert input data to float32
    input_img = input_img.astype(np.float32)
    # Create host buffer to receive data
    output = np.empty(output_size, dtype = np.float32)
    # Allocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.size * input_img.dtype.itemsize)
    d_output = cuda.mem_alloc(batch_size * output.size * output.dtype.itemsize)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # Execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # Synchronize threads
    stream.synchronize()
    # Return predictions
    return output

def get_testcase(path):
    im = Image.open(path)
    assert(im)
    return np.array(im).ravel()

def apply_mean(img, mean_path):
    parser = caffeparser.create_caffe_parser()
    # Parse mean
    meanblob = parser.parse_binary_proto(mean_path)
    parser.destroy()

    # Note: In TensorRT C++ no size is reqired, however you need it to cast the array
    mean = meanblob.get_data(INPUT_W * INPUT_H)

    data = np.empty([INPUT_H * INPUT_W])
    for i in range(INPUT_W * INPUT_H):
        data[i] = float(img[i]) - mean[i]

    meanblob.destroy()
    return data

def main():
    modelstream = API_to_model(WEIGHTS_PATH, 1)
    # Get random test case
    rand_file = randint(0, 9)
    img = get_testcase(DATA + str(rand_file) + '.pgm')
    print("Test case: " + str(rand_file))
    # Preprocess
    data = apply_mean(img, MEAN)
    # Load engine.
    runtime = trt.infer.create_infer_runtime(G_LOGGER)
    engine = runtime.deserialize_cuda_engine(modelstream.data(), modelstream.size(), None)
    if modelstream:
        modelstream.destroy()
    context = engine.create_execution_context()
    # Execute
    out = infer(context, data, OUTPUT_SIZE, 1)
    print("Prediction: " + str(np.argmax(out)))
    # Clean up
    context.destroy()
    engine.destroy()
    runtime.destroy()

if __name__ == "__main__":
    main()
