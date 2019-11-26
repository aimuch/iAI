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


# Script to dump TensorFlow weights in TRT v1 and v2 dump format.
# The V1 format is for TensorRT 4.0. The V2 format is for TensorRT 4.0 and later.

import sys
import struct
import argparse
try:
    import tensorflow as tf
    from tensorflow.python import pywrap_tensorflow
except ImportError as err:
    sys.stderr.write("""Error: Failed to import module ({})""".format(err))
    sys.exit()

parser = argparse.ArgumentParser(description='TensorFlow Weight Dumper')

parser.add_argument('-m', '--model', required=True, help='The checkpoint file basename, example basename(model.ckpt-766908.data-00000-of-00001) -> model.ckpt-766908')
parser.add_argument('-o', '--output', required=True, help='The weight file to dump all the weights to.')
parser.add_argument('-1', '--wtsv1', required=False, default=False, type=bool, help='Dump the weights in the wts v1.')

opt = parser.parse_args()

if opt.wtsv1:
    print "Outputting the trained weights in TensorRT's wts v1 format. This format is documented as:"
    print "Line 0: <number of buffers in the file>"
    print "Line 1-Num: [buffer name] [buffer type] [buffer size] <hex values>"
else:
    print "Outputting the trained weights in TensorRT's wts v2 format. This format is documented as:"
    print "Line 0: <number of buffers in the file>"
    print "Line 1-Num: [buffer name] [buffer type] [(buffer shape{e.g. (1, 2, 3)}] <buffer shaped size bytes of data>"

inputbase = opt.model
outputbase = opt.output

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def getTRTType(tensor):
    if tf.as_dtype(tensor.dtype) == tf.float32:
        return 0
    if tf.as_dtype(tensor.dtype) == tf.float16:
        return 1
    if tf.as_dtype(tensor.dtype) == tf.int8:
        return 2
    if tf.as_dtype(tensor.dtype) == tf.int32:
        return 3
    print("Tensor data type of %s is not supported in TensorRT"%(tensor.dtype))
    sys.exit();

try:
   # Open output file
    if opt.wtsv1:
        outputFileName = outputbase + ".wts"
    else:
        outputFileName = outputbase + ".wts2"
    outputFile = open(outputFileName, 'w')

    # read vars from checkpoint
    reader = pywrap_tensorflow.NewCheckpointReader(inputbase)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Record count of weights
    count = 0
    for key in sorted(var_to_shape_map):
        count += 1
    outputFile.write("%s\n"%(count))

    # Dump the weights in either v1 or v2 format
    for key in sorted(var_to_shape_map):
        tensor = reader.get_tensor(key)
        file_key = key.replace('/','_')
        typeOfElem = getTRTType(tensor)
        val = tensor.shape
        if opt.wtsv1:
            val = tensor.size
        print("%s %s %s "%(file_key, typeOfElem, val))
        flat_tensor = tensor.flatten()
        outputFile.write("%s 0 %s "%(file_key, val))
        if opt.wtsv1:
            for weight in flat_tensor:
                hexval = float_to_hex(float(weight))
                outputFile.write("%s "%(hexval[2:]))
        else:
            outputFile.write(flat_tensor.tobytes())
        outputFile.write("\n");
    outputFile.close()

except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
        print("It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")
        if ("Data loss" in str(e) and
                (any([e in inputbase for e in [".index", ".meta", ".data"]]))):
            proposed_file = ".".join(inputbase.split(".")[0:-1])
            v2_file_error_template = """
           It's likely that this is a V2 checkpoint and you need to provide the filename
           *prefix*.  Try removing the '.' and extension.  Try:
           inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposed_file))
