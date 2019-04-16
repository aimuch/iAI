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

from __future__ import print_function

try:
    import torch
    from torchvision import datasets, transforms
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have PyTorch and torchvision installed.
For installation instructions, see:
http://pytorch.org/""".format(err))

# TensorRT must be imported after any frameworks in the case where
# the framework has incorrect dependencies setup and is not updated
# to use the versions of libraries that TensorRT imports.
try:
    import tensorrt as trt
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH""".format(err))

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))

try:
    import argparse
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have pycuda and the example dependencies installed.
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]""".format(err))

import numpy as np
from random import randint

PARSER = argparse.ArgumentParser(description="Example of how to create a Caffe based TensorRT Engine and run inference")
PARSER.add_argument('datadir', help='Path to Python TensorRT data directory (realpath)')

ARGS = PARSER.parse_args()
DATA_DIR = ARGS.datadir

GLOGGER = trt.infer.ColorLogger(trt.infer.LogSeverity.ERROR)

#Create an Engine from a PyTorch model
def create_pytorch_engine(max_batch_size, builder, dt, model):
    network = builder.create_network()

    data = network.add_input("in", dt, (1, 28, 28))
    assert(data)

    #-------------
    conv1_w = model['conv1.weight'].cpu().numpy().reshape(-1)
    conv1_b = model['conv1.bias'].cpu().numpy().reshape(-1)
    conv1 = network.add_convolution(data, 20, (5,5),  conv1_w, conv1_b)
    assert(conv1)
    conv1.set_stride((1,1))

    #-------------
    pool1 = network.add_pooling(conv1.get_output(0), trt.infer.PoolingType.MAX, (2,2))
    assert(pool1)
    pool1.set_stride((2,2))

    #-------------
    conv2_w = model['conv2.weight'].cpu().numpy().reshape(-1)
    conv2_b = model['conv2.bias'].cpu().numpy().reshape(-1)
    conv2 = network.add_convolution(pool1.get_output(0), 50, (5,5), conv2_w, conv2_b)
    assert(conv2)
    conv2.set_stride((1,1))

    #-------------
    pool2 = network.add_pooling(conv2.get_output(0), trt.infer.PoolingType.MAX, (2,2))
    assert(pool2)
    pool2.set_stride((2,2))

    #-------------
    fc1_w = model['fc1.weight'].cpu().numpy().reshape(-1)
    fc1_b = model['fc1.bias'].cpu().numpy().reshape(-1)
    fc1 = network.add_fully_connected(pool2.get_output(0), 500, fc1_w, fc1_b)
    assert(fc1)

    #-------------
    relu1 = network.add_activation(fc1.get_output(0), trt.infer.ActivationType.RELU)
    assert(relu1)

    #-------------
    fc2_w = model['fc2.weight'].cpu().numpy().reshape(-1)
    fc2_b = model['fc2.bias'].cpu().numpy().reshape(-1)
    fc2 = network.add_fully_connected(relu1.get_output(0), 10, fc2_w, fc2_b)
    assert(fc2)

    #-------------
    # NOTE: Before release
    # Using log_softmax in training, cutting out log softmax here since no log softmax in TRT
    fc2.get_output(0).set_name("out")
    network.mark_output(fc2.get_output(0))


    builder.set_max_batch_size(max_batch_size)
    builder.set_max_workspace_size(1 << 20)

    engine = builder.build_cuda_engine(network)
    network.destroy()

    return engine

# Lambda to apply argmax to each result after inference to get prediction
argmax = lambda res: np.argmax(res.reshape(10))

# Using the PyTorch DataLoader to apply preprocessing and generate test cases
kwargs = {'num_workers': 1, 'pin_memory': True}

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp/mnist/data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=1,
    shuffle=True,
    **kwargs
    )

def generate_cases(num):
    '''
    Generate num cases from the PyTorch DataLoader
    '''
    cases = []
    labels = []
    for index, (case, target) in enumerate(test_loader):
        cases.append(case.numpy().reshape(1, 28, 28))
        labels.append(target.numpy()[0])
        if index >= num:
            return cases, labels

def main():
    # Load pretrained PyTorch model
    model = torch.load(DATA_DIR + "/mnist/trained_lenet5_mnist.pyt")

    # Build an engine from PyTorch
    builder = trt.infer.create_infer_builder(GLOGGER)
    engine = create_pytorch_engine(10, builder, trt.infer.DataType.FLOAT, model)

    # Create a Lite Engine from the engine
    mnist_engine = trt.lite.Engine(engine_stream=engine.serialize(), # Use a serialized engine to create a Lite Engine
                                   max_batch_size=10,                # Max batch size
                                   postprocessors={"out":argmax})    # Post process data

    # Destroy the old engine
    engine.destroy()
    # Generate cases
    data, target = generate_cases(10)
    # Run inference on our generated cases
    results = mnist_engine.infer(data)[0]

    # Validate results
    correct = 0
    print ("[LABEL] | [RESULT]")
    for l in range(len(target)):
        print ("   {}    |    {}   ".format(target[l], results[l]))
        if target[l] == results[l]:
            correct += 1
    print ("Inference: {:.2f}% Correct".format((correct / float(len(target))) * 100))

if __name__ == "__main__":
    main()
