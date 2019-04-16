#!/usr/bin/env python3

import os
import ctypes
import time
import sys
import argparse

import numpy as np
from PIL import Image
import tensorrt as trt

import utils.inference as inference_utils # TRT/TF inference wrappers
import utils.model as model_utils # UFF conversion
import utils.boxes as boxes_utils # Drawing bounding boxes
import utils.coco as coco_utils # COCO dataset descriptors
from utils.paths import PATHS # Path management


# COCO label list
COCO_LABELS = coco_utils.COCO_CLASSES_LIST

# Model used for inference
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'

# Confidence threshold for drawing bounding box
VISUALIZATION_THRESHOLD = 0.5

# Precision command line argument -> TRT Engine datatype
TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}

# Layout of TensorRT network output metadata
TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}


def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    """Fetches prediction field from prediction byte array.

    After TensorRT inference, prediction data is saved in
    byte array and returned by object detection network.
    This byte array contains several pieces of data about
    prediction - we call one such piece a prediction field.
    The prediction fields layout is described in TRT_PREDICTION_LAYOUT.

    This function, given prediction byte array returned by network,
    staring index of given prediction and field name of interest,
    returns prediction field data corresponding to given arguments.

    Args:
        field_name (str): field of interest, one of keys of TRT_PREDICTION_LAYOUT
        detection_out (array): object detection network output
        pred_start_idx (int): start index of prediction of interest in detection_out

    Returns:
        Prediction field corresponding to given data.
    """
    return detection_out[pred_start_idx + TRT_PREDICTION_LAYOUT[field_name]]

def analyze_prediction(detection_out, pred_start_idx, img_pil):
    image_id = int(fetch_prediction_field("image_id", detection_out, pred_start_idx))
    label = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field("confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)
    if confidence > VISUALIZATION_THRESHOLD:
        class_name = COCO_LABELS[label]
        confidence_percentage = "{0:.0%}".format(confidence)
        print("Detected {} with confidence {}".format(
            class_name, confidence_percentage))
        boxes_utils.draw_bounding_boxes_on_image(
            img_pil, np.array([[ymin, xmin, ymax, xmax]]),
            display_str_list=["{}: {}".format(
                class_name, confidence_percentage)],
            color=coco_utils.COCO_COLORS[label]
        )

def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    parser = argparse.ArgumentParser(description='Run object detection inference on input image.')
    parser.add_argument('input_img_path', metavar='INPUT_IMG_PATH',
        help='an image file to run inference on')
    parser.add_argument('-p', '--precision', type=int, choices=[32, 16], default=32,
        help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size', type=int, default=1,
        help='max TensorRT engine batch size')
    parser.add_argument('-w', '--workspace_dir',
        help='sample workspace directory')
    parser.add_argument('-fc', '--flatten_concat',
        help='path of built FlattenConcat plugin')

    # Parse arguments passed
    args = parser.parse_args()

    # Set FlattenConcat TRT plugin path and
    # workspace dir path if passed by user
    if args.flatten_concat:
        PATHS.set_flatten_concat_plugin_path(args.flatten_concat)
    if args.workspace_dir:
        PATHS.set_workspace_dir_path(args.workspace_dir)
    if not os.path.exists(PATHS.get_workspace_dir_path()):
        os.makedirs(PATHS.get_workspace_dir_path())

    # Verify Paths after adjustments. This also exits script if verification fails
    PATHS.verify_all_paths()

    # Fetch TensorRT engine path and datatype
    trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.precision]
    trt_engine_path = PATHS.get_engine_path(trt_engine_datatype,
        args.max_batch_size)
    if not os.path.exists(os.path.dirname(trt_engine_path)):
        os.makedirs(os.path.dirname(trt_engine_path))

    parsed = {
        'input_img_path': args.input_img_path,
        'max_batch_size': args.max_batch_size,
        'trt_engine_datatype': trt_engine_datatype,
        'trt_engine_path': trt_engine_path
    }
    return parsed

def main():
    # Parse command line arguments
    parsed = parse_commandline_arguments()

    # Loading FlattenConcat plugin library using CDLL has a side
    # effect of loading FlattenConcat plugin into internal TensorRT
    # PluginRegistry data structure. This will be needed when parsing
    # network into UFF, since some operations will need to use this plugin
    try:
        ctypes.CDLL(PATHS.get_flatten_concat_plugin_path())
    except:
        print(
            "Error: {}\n{}\n{}".format(
                "Could not find {}".format(PATHS.get_flatten_concat_plugin_path()),
                "Make sure you have compiled FlattenConcat custom plugin layer",
                "For more details, check README.md"
            )
        )
        sys.exit(1)

    # Fetch .uff model path, convert from .pb
    # if needed, using prepare_ssd_model
    ssd_model_uff_path = PATHS.get_model_uff_path(MODEL_NAME)
    if not os.path.exists(ssd_model_uff_path):
        model_utils.prepare_ssd_model(MODEL_NAME)

    # Set up all TensorRT data structures needed for inference
    trt_inference_wrapper = inference_utils.TRTInference(
        parsed['trt_engine_path'], ssd_model_uff_path,
        trt_engine_datatype=parsed['trt_engine_datatype'],
        batch_size=parsed['max_batch_size'])

    # Start measuring time
    inference_start_time = time.time()

    # Get TensorRT SSD model output
    detection_out, keep_count_out = \
        trt_inference_wrapper.infer(parsed['input_img_path'])

    # Make PIL.Image for drawing bounding boxes and
    # let analyze_prediction() draw them based on model output
    img_pil = Image.open(parsed['input_img_path'])
    prediction_fields = len(TRT_PREDICTION_LAYOUT)
    for det in range(int(keep_count_out[0])):
        analyze_prediction(detection_out, det * prediction_fields, img_pil)

    # Output total [img load + inference + drawing bboxes] time
    print("Total time taken for one image: {} ms\n".format(
        int(round((time.time() - inference_start_time) * 1000))))

    # Save output image and output path
    inferred_image_path = os.path.join(PATHS.get_sample_root(), "image_inferred.jpg")
    img_pil.save(inferred_image_path)
    print("Saved output image to: {}".format(inferred_image_path))


if __name__ == '__main__':
    main()
