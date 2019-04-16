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
    import tensorrt
    from tensorrt.parsers import caffeparser
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


#Get the mean image from the caffe binaryproto file
parser = caffeparser.create_caffe_parser()
mean_blob = parser.parse_binary_proto(DATA_DIR + "/mnist/mnist_mean.binaryproto")
# parser.destroy()
MEAN = mean_blob.get_data(28 * 28)

def sub_mean(img):
    '''
    A function to subtract the mean image from a test case
    Will be registered in the Lite Engine preprocessor table
    to be applied to each input case
    '''
    img = img.ravel()
    data = np.empty(len(img))
    for i in range(len(img)):
        data[i] = np.float32((img[i]) - MEAN[i])
    return data.reshape(1, 28, 28)

#Lamba to apply argmax to each result after inference to get prediction
argmax = lambda res: np.argmax(res.reshape(10))

#Create an engine from a caffe model
caffe_engine = tensorrt.lite.Engine(framework="c1",                                   #Source framework
                                    deployfile=DATA_DIR + "/mnist/mnist.prototxt",    #Deploy file
                                    modelfile=DATA_DIR + "/mnist/mnist.caffemodel",   #Model File
                                    max_batch_size=10,                                #Max number of images to be processed at a time
                                    logger_severity=tensorrt.infer.LogSeverity.ERROR, #Suppress debugging info
                                    input_nodes={"data":(1,28,28)},                   #Input layers
                                    output_nodes=["prob"])                            #Output layers

#Save engine and delete
caffe_engine.save("/tmp/caffe_lenet5_mnist.plan")
caffe_engine.__del__

mnist_engine = tensorrt.lite.Engine(PLAN="/tmp/caffe_lenet5_mnist.plan",
                                    max_batch_size=10,
                                    preprocessors={"data":sub_mean},
                                    postprocessors={"prob":argmax})

def generate_cases(num):
    '''
    Generate a list of data to process and answers to compare to
    '''
    cases = []
    labels = []
    for c in range(num):
        rand_file = randint(0, 9)
        im = Image.open(DATA_DIR + "/mnist/" + str(rand_file) + ".pgm")
        arr = np.array(im).reshape(1,28,28) #Make the image CHANNEL x HEIGHT x WIDTH
        cases.append(arr) #Append the image to list of images to process
        labels.append(rand_file) #Append the correct answer to compare later
    return cases, labels


def main():
    #Generate cases
    data, target = generate_cases(10)
    #Run inference on our generated cases
    results = mnist_engine.infer(data)[0]

    #Validate results
    correct = 0
    print ("[LABEL] | [RESULT]")
    for l in range(len(target)):
        print ("   {}    |    {}   ".format(target[l], results[l]))
        if target[l] == results[l]:
            correct += 1
    print ("Inference: {:.2f}% Correct".format((correct / len(target)) * 100))

if __name__ == "__main__":
    main()
