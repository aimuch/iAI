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

# VOC dataset utility functions
import numpy as np


VOC_CLASSES_LIST = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

VOC_CLASSES_SET = set(VOC_CLASSES_LIST)

VOC_CLASS_ID = {
    cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES_LIST)
}

# Random RGB colors for each class (useful for drawing bounding boxes)
VOC_COLORS = \
    np.random.uniform(0, 255, size=(len(VOC_CLASSES_LIST), 3)).astype(np.uint8)


def convert_coco_to_voc(label):
    """Converts COCO class name to VOC class name, if possible.

    COCO classes are a superset of VOC classes, but
    some classes have different names (e.g. airplane
    in COCO is aeroplane in VOC). This function gets
    COCO label and converts it to VOC label,
    if conversion is needed.

    Args:
        label (str): COCO label
    Returns:
        str: VOC label corresponding to given label if such exists,
            otherwise returns original label
    """
    COCO_VOC_DICT = {
        'airplane': 'aeroplane',
        'motorcycle': 'motorbike',
        'dining table': 'diningtable',
        'potted plant': 'pottedplant',
        'couch': 'sofa',
        'tv': 'tvmonitor'
    }
    if label in COCO_VOC_DICT:
        return COCO_VOC_DICT[label]
    else:
        return label

def coco_label_to_voc_label(label):
    """Returns VOC label corresponding to given COCO label.

    COCO classes are superset of VOC classes, this function
    returns label corresponding to given COCO class label
    or None if such label doesn't exist.

    Args:
        label (str): COCO class label
    Returns:
        str: VOC label corresponding to given label or None
    """
    label = convert_coco_to_voc(label)
    if label in VOC_CLASSES_SET:
        return label
    else:
        return None

def is_voc_label(label):
    """Returns boolean which tells if given label is VOC label.

    Args:
        label (str): object label
    Returns:
        bool: is given label a VOC class label
    """
    return label in VOC_CLASSES_SET

def get_voc_label_color(label):
    """Returns color corresponding to given VOC label, or None.

    Args:
        label (str): object label
    Returns:
        np.array: RGB color described in 3-element np.array
    """
    if not is_voc_label(label):
        return None
    else:
        return VOC_COLORS[VOC_CLASS_ID[label]]
