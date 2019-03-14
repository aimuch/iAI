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
    from tensorrt import parsers
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH""".format(err))

class Profiler(trt.infer.Profiler):
    """
    Example Implimentation of a Profiler
    Is identical to the Profiler class in trt.infer so it is possible
    to just use that instead of implementing this if further
    functionality is not needed
    """
    def __init__(self, timing_iter):
        trt.infer.Profiler.__init__(self)
        self.timing_iterations = timing_iter
        self.profile = []

    def report_layer_time(self, layerName, ms):
        record = next((r for r in self.profile if r[0] == layerName), (None, None))
        if record == (None, None):
            self.profile.append((layerName, ms))
        else:
            self.profile[self.profile.index(record)] = (record[0], record[1] + ms)

    def print_layer_times(self):
        totalTime = 0
        for i in range(len(self.profile)):
            print("{:40.40} {:4.3f}ms".format(self.profile[i][0], self.profile[i][1] / self.timing_iterations))
            totalTime += self.profile[i][1]
        print("Time over all layers: {:4.3f}".format(totalTime / self.timing_iterations))


BATCH_SIZE = 4
TIMING_INTERATIONS = 1000

G_PROFILER = Profiler(TIMING_INTERATIONS)
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
INPUT_LAYERS = ["data"]
OUTPUT_LAYERS = ['prob']
OUTPUT_SIZE = 10

PARSER = argparse.ArgumentParser(description="Example of how to create a Caffe based TensorRT Engine and profile inference")
PARSER.add_argument('datadir', help='Path to Python TensorRT data directory (realpath)')

DATA_DIR = PARSER.parse_args().datadir

MODEL_PROTOTXT = DATA_DIR + "/googlenet/googlenet.prototxt"
CAFFEMODEL = DATA_DIR + "/googlenet/googlenet.caffemodel"

#Run inference on device
def time_inference(engine, batch_size):
    assert(engine.get_nb_bindings() == 2)

    input_index = engine.get_binding_index(INPUT_LAYERS[0])
    output_index = engine.get_binding_index(OUTPUT_LAYERS[0])

    input_dim = engine.get_binding_dimensions(input_index).to_DimsCHW()
    output_dim = engine.get_binding_dimensions(output_index).to_DimsCHW()

    insize = batch_size * input_dim.C() * input_dim.H() * input_dim.W() * 4
    outsize = batch_size * output_dim.C() * output_dim.H() * output_dim.W() * 4

    d_input = cuda.mem_alloc(insize)
    d_output = cuda.mem_alloc(outsize)

    bindings = [int(d_input), int(d_output)]

    context = engine.create_execution_context()
    context.set_profiler(G_PROFILER)

    cuda.memset_d32(d_input, 0, insize // 4)

    for i in range(TIMING_INTERATIONS):
        context.execute(batch_size, bindings)

    context.destroy()
    return


def main():
    path = dir_path = os.path.dirname(os.path.realpath(__file__))

    print("Building and running GPU inference for GoogleNet, N=4")
    #Convert caffe model to TensorRT engine
    engine = trt.utils.caffe_to_trt_engine(G_LOGGER,
        MODEL_PROTOTXT,
        CAFFEMODEL,
        10,
        16 << 20,
        OUTPUT_LAYERS,
        trt.infer.DataType.FLOAT)

    runtime = trt.infer.create_infer_runtime(G_LOGGER)

    print("Bindings after deserializing")
    for bi in range(engine.get_nb_bindings()):
        if engine.binding_is_input(bi) == True:
            print("Binding " + str(bi) + " (" + engine.get_binding_name(bi) + "): Input")
        else:
            print("Binding " + str(bi) + " (" + engine.get_binding_name(bi) + "): Output")

    time_inference(engine, BATCH_SIZE)

    engine.destroy()
    runtime.destroy()

    G_PROFILER.print_layer_times()

    print("Done")

    return

if __name__ == "__main__":
    main()
