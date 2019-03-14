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

#... (1) First we import the TensorRT
try:
    import tensorrt as trt
    from tensorrt import parsers
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH""".format(err))


#... (2) Now we import some necessary python packages
from random import randint
import numpy as np
import sys

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
    raise ImportError("""ERROR: failed to import module ({})
Please make sure you have pycuda and the example dependencies installed.
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]
""".format(err))


#... (3) Create a Logger. The argument specifies the default verbosity level.
#...     trt.infer.LogSeverity.ERROR will report only ERRORS, while INFO will
#...     report all ERROR, WARNING and INFO messages.
#...     For example:
#...     G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.WARNING)
#...     G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

#... (4) Define some constants
INPUT_LAYERS = ["data"]
OUTPUT_LAYERS = ['prob']
INPUT_H = 28
INPUT_W =  28
OUTPUT_SIZE = 10

PARSER = argparse.ArgumentParser(description="Example of how to create a Caffe based TensorRT Engine and run inference")
PARSER.add_argument('datadir', help='Path to Python TensorRT data directory (realpath)')

ARGS = PARSER.parse_args()
DATA_DIR = ARGS.datadir

#... (5) Define the paths the model data is located
MODEL_PROTOTXT = DATA_DIR + '/mnist/mnist.prototxt'
CAFFE_MODEL =  DATA_DIR + '/mnist/mnist.caffemodel'
DATA =  DATA_DIR + '/mnist/'
IMAGE_MEAN =  DATA_DIR + '/mnist/mnist_mean.binaryproto'



def get_testcase(path):
    im = Image.open(path)
    assert(im)
    arr = np.array(im)
    #make array 1D
    img = arr.ravel()
    return img


def infer(context, input_img, output_size, batch_size):
    #load engine
    engine = context.get_engine()
    assert(engine.get_nb_bindings() == 2)
    #convert input data to Float32
    input_img = input_img.astype(np.float32)
    #create output array to receive data
    output = np.empty(output_size, dtype = np.float32)

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

    #synchronize threads
    stream.synchronize()

    #return predictions
    return output


def main():
    deploy_file = MODEL_PROTOTXT
    model_file = CAFFE_MODEL
    max_batch_size = 1
    max_workspace_size = 1 << 20
    output_layers = OUTPUT_LAYERS
    datatype = trt.infer.DataType.FLOAT

    #parse_wrapper(deploy_file, model_file, max_batch_size, max_workspace_size, output_layers, datatype)

    #... The count continues from the outside the main function definition.
    #... (6) create the builder
    builder = trt.infer.create_infer_builder(G_LOGGER)

    #... (7) Create the network
    network = builder.create_network()

    #... (8) Create Parser
    parser = parsers.caffeparser.create_caffe_parser()

    #...Randomly pick up the test case
    rand_file = randint(0, 9)
    img = get_testcase(DATA + str(rand_file) + '.pgm')
    print("Test case: " + str(rand_file))
    #...End Randomly pick up the test case

    #... (9) Normalize the data
    mean_blob = parser.parse_binary_proto(IMAGE_MEAN)
    mean = mean_blob.get_data(INPUT_W * INPUT_H)

    normalized_data = np.empty([INPUT_H * INPUT_W])
    for i in range(INPUT_W * INPUT_H):
        normalized_data[i] = float(img[i]) - mean[i]

    mean_blob.destroy()
    #...End Normalization

    #... (10) Parse the caffe network and weights, create TRT network
    #...     The output is really here a populated network, but the blob_name_to_tensor is the table of name to ITensor
    blob_name_to_tensor = parser.parse(deploy_file, model_file, network, datatype)


    #------- This is really ap to here

    G_LOGGER.log(trt.infer.LogSeverity.INFO, "Parsing caffe model {}, {}".format(deploy_file, model_file))

    '''
    #...Debug print of the Input Dimensions
    input_dimensions = {}

    for i in range(network.get_nb_inputs()):
        dims = network.get_input(i).get_dimensions().to_DimsCHW()
        G_LOGGER.log(trt.infer.LogSeverity.INFO, "Input \"{}\":{}x{}x{}".format(network.get_input(i).get_name(), dims.C(), dims.H(), dims.W()))
        input_dimensions[network.get_input(i).get_name()] = network.get_input(i).get_dimensions().to_DimsCHW()

    print(type(output_layers) )
    '''

    #...Execute Inference

    #...Mark Outputs
    #... (1)
    for l in output_layers:
        G_LOGGER.log(trt.infer.LogSeverity.INFO, "Marking " + l + " as output layer")
        t = blob_name_to_tensor.find(l)
        try:
            assert(t)
        except AssertionError:
            G_LOGGER.log(trt.infer.LogSeverity.ERROR, "Failed to find output layer {}".format(l))
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb) # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            raise AssertionError('Caffe parsing failed on line {} in statement {}'.format(line, text))

        layer = network.mark_output(t)


    '''
    #...Debug print of the Output Dimensions
    for i in range(network.get_nb_outputs()):
        dims = network.get_output(i).get_dimensions().to_DimsCHW()
        G_LOGGER.log(trt.infer.LogSeverity.INFO, "Output \"{}\":{}x{}x{}".format(network.get_output(i).get_name(), dims.C(), dims.H(), dims.W()))
    '''

    builder.set_max_batch_size(max_batch_size)
    builder.set_max_workspace_size(max_workspace_size)

    engine = builder.build_cuda_engine(network)

    context = engine.create_execution_context()

    out = infer(context, normalized_data, OUTPUT_SIZE, 1)

    print("Prediction: " + str(np.argmax(out)))

    context.destroy()
    network.destroy()
    parser.destroy()
    builder.destroy()




if __name__ == "__main__":
    main()


