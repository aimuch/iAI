#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
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

# This sample uses a Caffe model along with a custom plugin to create a TensorRT engine.
from random import randint
from PIL import Image
import numpy as np
import tempfile

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

try:
    from build import fcplugin
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please build the FullyConnected sample plugin.
For more information, see the included README.md
Note that Python 2 requires the presence of `__init__.py` in the build folder""".format(err))

# Allows us to import from common.
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Define some global constants about the model.
class ModelData(object):
    INPUT_NAME = "input"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SHAPE = (10, )
    DTYPE = trt.float32

# Uses a parser to retrieve mean data from a binary_proto.
def retrieve_mean(mean_proto):
    with trt.CaffeParser() as parser:
        return parser.parse_binary_proto(mean_proto)

# Create the parser's plugin factory. The factory is global because it has
# to be destroyed after the engine is destroyed.
fc_factory = fcplugin.FCPluginFactory()

# For more information on TRT basics, refer to the introductory parser samples.
def build_engine(deploy_file, model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        builder.max_workspace_size = common.GiB(1)

        # Set the parser's plugin factory. Note that we bind the factory to a reference so
        # that we can destroy it later. (parser.plugin_factory_ext is a write-only attribute)
        parser.plugin_factory_ext = fc_factory

        # Parse the model and build the engine.
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        return builder.build_cuda_engine(network)

# Tries to load an engine from the provided engine_path, or builds and saves an engine to the engine_path.
def get_engine(deploy_file, model_file, engine_path):
    try:
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            # Note that we have to provide the plugin factory when deserializing an engine built with an IPlugin or IPluginExt.
            return runtime.deserialize_cuda_engine(f.read(), fc_factory)
    except:
        # Fallback to building an engine if the engine cannot be loaded for any reason.
        engine = build_engine(deploy_file, model_file)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        return engine

# Loads a test case into the provided pagelocked_buffer.
def load_normalized_test_case(data_paths, mean):
    case_num = randint(0, 9)
    [test_case_path] = common.locate_files(data_paths, [str(case_num) + ".pgm"])
    # Flatten the image into a 1D array, and normalize.
    img = np.array(Image.open(test_case_path)).ravel() - mean
    return img, case_num

def main():
    # Get data files for the model.
    data_paths, [deploy_file, model_file, mean_proto] = common.find_sample_data(description="Runs an MNIST network using a Caffe model file", subfolder="mnist", find_files=["mnist.prototxt", "mnist.caffemodel", "mnist_mean.binaryproto"])

    # Cache the engine in a temporary directory.
    engine_path = os.path.join(tempfile.gettempdir(), "mnist.engine")
    with get_engine(deploy_file, model_file, engine_path) as engine, engine.create_execution_context() as context:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        mean = retrieve_mean(mean_proto)
        # For more information on performing inference, refer to the introductory samples.
        inputs[0].host, case_num = load_normalized_test_case(data_paths, mean)
        # The common.do_inference function will return a list of outputs - we only have one in this case.
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = np.argmax(output)
        print("Test Case: " + str(case_num))
        print("Prediction: " + str(pred))

    # After the engine is destroyed, we destroy the plugin. This function is exposed through the binding code in plugin/pyFullyConnected.cpp.
    fc_factory.destroy_plugin()

if __name__ == "__main__":
    main()
