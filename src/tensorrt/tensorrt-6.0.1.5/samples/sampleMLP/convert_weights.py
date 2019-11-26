#!/usr/bin/python
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


# Script to convert from TensorFlow weights to TensorRT weights for multilayer perceptron sample.
# Change the remap to properly remap the weights to the name from your trained model
# to the sample expected format.

import sys
import struct
import argparse

try:
    from tensorflow.python import pywrap_tensorflow as pyTF
except ImportError as err:
    sys.stderr.write("""Error: Failed to import module ({})""".format(err))
    sys.exit()

parser = argparse.ArgumentParser(description='TensorFlow to TensorRT Weight Dumper')

parser.add_argument('-m', '--model', required=True, help='The checkpoint file basename, example basename(model.ckpt-766908.data-00000-of-00001) -> model.ckpt-766908')
parser.add_argument('-o', '--output', required=True, help='The weight file to dump all the weights to.')

opt = parser.parse_args()

print "Outputting the trained weights in TensorRT's wts v2 format. This format is documented as:"
print "Line 0: <number of buffers in the file>"
print "Line 1-Num: [buffer name] [buffer type] [(buffer shape{e.g. (1, 2, 3)}] <buffer shaped size bytes of data>"

inputbase = opt.model
outputbase = opt.output

# This dictionary translates from the TF weight names to the weight names expected 
# by the sampleMLP sample. This is the location that needs to be changed if training
# something other than what is specified in README.txt.
remap = {
    'Variable': 'hiddenWeights0',
    'Variable_1': 'hiddenWeights1',
    'Variable_2': 'outputWeights',
    'Variable_3': 'hiddenBias0',
    'Variable_4': 'hiddenBias1',
    'Variable_5': 'outputBias'
}

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

try:
   reader = pyTF.NewCheckpointReader(inputbase)
   tensorDict = reader.get_variable_to_shape_map()
   outputFileName = outputbase + ".wts2"
   outputFile = open(outputFileName, 'w')
   count = 0

   for key in sorted(tensorDict):
       # Don't count weights that aren't used for inferencing.
       if ("Adam" in key or "power" in key):
           continue
       count += 1
   outputFile.write("%s\n"%(count))

   for key in sorted(tensorDict):
       # In order to save space, we don't dump weights that aren't required.
       if ("Adam" in key or "power" in key):
           continue
       tensor = reader.get_tensor(key)
       file_key = remap[key.replace('/','_')]
       val = tensor.shape
       print("%s 0 %s "%(file_key, val))
       flat_tensor = tensor.flatten()
       outputFile.write("%s 0 %s "%(file_key, val))
       outputFile.write(flat_tensor.tobytes())
       outputFile.write("\n");
   outputFile.close()

except Exception as error:
    print(str(error))
