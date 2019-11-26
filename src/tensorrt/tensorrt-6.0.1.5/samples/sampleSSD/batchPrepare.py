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

import argparse
import numpy as np
import sys
import os
import glob
import shutil
import struct
from random import shuffle

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))

height = 300
width = 300
NUM_BATCHES = 0
NUM_PER_BATCH = 1
NUM_CALIBRATION_IMAGES = 500

parser = argparse.ArgumentParser()
parser.add_argument('--inDir', required=True, help='Input directory')
parser.add_argument('--outDir', required=True, help='Output directory')

args = parser.parse_args()

CALIBRATION_DATASET_LOC = args.inDir + '/*.jpg'


# images to test
imgs = []
print("Location of dataset = " + CALIBRATION_DATASET_LOC)
imgs = glob.glob(CALIBRATION_DATASET_LOC)
shuffle(imgs)
imgs = imgs[:NUM_CALIBRATION_IMAGES]
NUM_BATCHES = NUM_CALIBRATION_IMAGES // NUM_PER_BATCH + (NUM_CALIBRATION_IMAGES % NUM_PER_BATCH > 0)

print("Total number of images = " + str(len(imgs)))
print("NUM_PER_BATCH = " + str(NUM_PER_BATCH))
print("NUM_BATCHES = " + str(NUM_BATCHES))

# output
outDir  = args.outDir+"/batches"

if os.path.exists(outDir):
	os.system("rm " + outDir +"/*")

# prepare output
if not os.path.exists(outDir):
	os.makedirs(outDir)

for i in range(NUM_CALIBRATION_IMAGES):
	os.system("convert "+imgs[i]+" -resize "+str(height)+"x"+str(width)+"! "+outDir+"/"+str(i)+".ppm")

CALIBRATION_DATASET_LOC= outDir + '/*.ppm'
imgs = glob.glob(CALIBRATION_DATASET_LOC)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
img = 0
for i in range(NUM_BATCHES):
	batchfile = outDir + "/batch_calibration" + str(i) + ".batch"
	batchlistfile = outDir + "/batch_calibration" + str(i) + ".list"
	batchlist = open(batchlistfile,'a')
	batch = np.zeros(shape=(NUM_PER_BATCH, 3, height, width), dtype = np.float32)
	for j in range(NUM_PER_BATCH):
		batchlist.write(os.path.basename(imgs[img]) + '\n')
		im = Image.open(imgs[img]).resize((width,height), Image.NEAREST)
		in_ = np.array(im, dtype=np.float32, order='C')
		in_ = in_[:,:,::-1]
		in_-= np.array((104.0, 117.0, 123.0))
		in_ = in_.transpose((2,0,1))
		batch[j] = in_
		img += 1

	# save
	batch.tofile(batchfile)
	batchlist.close()

	# Prepend batch shape information
	ba = bytearray(struct.pack("4i", batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]))

	with open(batchfile, 'rb+') as f:
		content = f.read()
		f.seek(0,0)
		f.write(ba)
		f.write(content)

os.system("rm " + outDir +"/*.ppm")
