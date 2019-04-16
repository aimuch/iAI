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
from __future__ import division
from __future__ import print_function
import os
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
    import pycuda.autoinit
    import argparse
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have pycuda and the example dependencies installed.
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]""".format(err))

try:
    import uff
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the UFF Toolkit installed.""".format(err))

try:
    import tensorrt as trt
    from tensorrt.parsers import uffparser
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH""".format(err))

import lenet5


MAX_WORKSPACE = 1 << 30
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

INPUT_W = 28
INPUT_H = 28
OUTPUT_SIZE = 10

MAX_BATCHSIZE = 1
ITERATIONS = 10


# API CHANGE: Try to generalize into a utils function
#Run inference on device
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
    path = os.path.dirname(os.path.realpath(__file__))

    tf_model = lenet5.learn()

    uff_model = uff.from_tensorflow(tf_model, ["fc2/Relu"])
	#Convert Tensorflow model to TensorRT model
    parser = uffparser.create_uff_parser()
    parser.register_input("Placeholder", (1, 28, 28), 0)
    parser.register_output("fc2/Relu")

    engine = trt.utils.uff_to_trt_engine(G_LOGGER,
                                              uff_model,
                                              parser,
                                              MAX_BATCHSIZE,
                                              MAX_WORKSPACE)

    assert(engine)

    parser.destroy()
    context = engine.create_execution_context()

    print("\n| TEST CASE | PREDICTION |")
    for i in range(ITERATIONS):
        img, label = lenet5.get_testcase()
        img = img[0]
        label = label[0]
        out = infer(context, img, 1)
        print("|-----------|------------|")
        print("|     " + str(label) + "     |      " + str(np.argmax(out)) + "     |")



if __name__ == "__main__":
    main()
