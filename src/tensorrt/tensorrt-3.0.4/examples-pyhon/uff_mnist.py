#
# Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
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
from __future__ import division
from __future__ import print_function
import os
import sys
from random import randint
import numpy as np

try:
    from PIL import Image
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
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
    import tensorrt as trt
    from tensorrt.parsers import uffparser
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({}) 
Please make sure you have the TensorRT Library installed 
and accessible in your LD_LIBRARY_PATH
""".format(err))
    exit(1) 

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
MAX_WORKSPACE = 1 << 30
MAX_BATCHSIZE = 1

PARSER = argparse.ArgumentParser(description="Example of how to create a Caffe based TensorRT Engine and run inference")
PARSER.add_argument('datadir', help='Path to Python TensorRT data directory (realpath)')

ARGS = PARSER.parse_args()
DATA_DIR = ARGS.datadir

DATA=DATA_DIR + '/mnist/'
MODEL=DATA_DIR + '/mnist/lenet5.uff'
# API CHANGE: Try to generalize into a utils function
#Run inference on device
def infer(engine, input_img, batch_size):
    #load engine
    context = engine.create_execution_context()
    assert(engine.get_nb_bindings() == 2)
    #create output array to receive data
    dims = engine.get_binding_dimensions(1).to_DimsCHW()
    elt_count = dims.C() * dims.H() * dims.W() * batch_size
    #Allocate pagelocked memory
    output = cuda.pagelocked_empty(elt_count, dtype = np.float32)

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

def get_testcase(path):
    im = Image.open(path)
    assert(im)
    arr = np.array(im)
    #make array 1D
    img = arr.ravel()
    return img

#Also prints case to console 
def normalize(data):
    #allocate pagelocked memory
    norm_data = cuda.pagelocked_empty(data.shape, np.float32)
    print("\n\n\n---------------------------", "\n")
    for i in range(len(data)):
        print(" .:-=+*#%@"[data[i] // 26] + ("\n" if ((i + 1) % 28 == 0) else ""), end="");
        norm_data[i] = 1.0 - data[i] / 255.0
    print("\n")
    return norm_data


def main():
    path = dir_path = os.path.dirname(os.path.realpath(__file__))

    #Convert uff model to TensorRT model
    parser = uffparser.create_uff_parser()
    parser.register_input("Input_0", (1, 28, 28), 0)
    parser.register_output("Binary_3")

    engine = trt.utils.uff_file_to_trt_engine(G_LOGGER,
                                              MODEL,
                                              parser,
                                              MAX_BATCHSIZE,
                                              MAX_WORKSPACE,
                                              trt.infer.DataType.FLOAT)

    assert(engine)

    # parser.destroy()

    rand_file = randint(0, 9)
    img = get_testcase(DATA + str(rand_file) + '.pgm')
    data = normalize(img)

    print("Test case: " + str(rand_file))

    out = infer(engine, data, 1)

    print("Prediction: " + str(np.argmax(out)))


if __name__ == "__main__":
    main()
