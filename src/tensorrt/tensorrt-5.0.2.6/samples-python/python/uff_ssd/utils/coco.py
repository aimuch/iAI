# COCO dataset utility functions
import numpy as np


COCO_CLASSES_LIST = [
    'unlabeled',
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

COCO_CLASSES_SET = set(COCO_CLASSES_LIST)

COCO_CLASS_ID = {
    cls_name: idx for idx, cls_name in enumerate(COCO_CLASSES_LIST)
}

# Random RGB colors for each class (useful for drawing bounding boxes)
COCO_COLORS = \
    np.random.uniform(0, 255, size=(len(COCO_CLASSES_LIST), 3)).astype(np.uint8)


def is_coco_label(label):
    """Returns boolean which tells if given label is COCO label.

    Args:
        label (str): object label
    Returns:
        bool: is given label a COCO class label
    """
    return label in COCO_CLASSES_SET

def get_coco_label_color(label):
    """Returns color corresponding to given COCO label, or None.

    Args:
        label (str): object label
    Returns:
        np.array: RGB color described in 3-element np.array
    """
    if not is_coco_label(label):
        return None
    else:
        return COCO_COLORS[COCO_CLASS_ID[label]]
