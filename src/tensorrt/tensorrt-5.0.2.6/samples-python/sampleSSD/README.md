# NVIDIA TensorRT Sample "sampleSSD"

This example is based on the following paper, SSD: Single Shot MultiBox 
Detector (https://arxiv.org/abs/1512.02325). The SSD network performs the 
task of object detection and localization in a single forward pass of the 
network. This network is trained on VGG network using PASCAL VOC 2007+ 2012 
datasets. This sample uses the dataset from here: https://github.com/weiliu89/caffe/tree/ssd

## How to get caffe model

* Download models_VGGNet_VOC0712_SSD_300x300.tar.gz to ~/Downloads with 
the link provided by the author of SSD: https://drive.google.com/file/d/0BzKzrI_SkD1_WVVTSmQxU0dVRzA/view
* tar xvf ~/Downloads/models_VGGNet_VOC0712_SSD_300x300.tar.gz --exclude=*prototxt --exclude=*.py --strip-components=4
* MD5 hash commands:
  md5sum models_VGGNet_VOC0712_SSD_300x300.tar.gz
  Expected MD5 hash:
  9a795fc161fff2e8f3aed07f4d488faf  models_VGGNet_VOC0712_SSD_300x300.tar.gz

* mv ~/Downloads/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel <TensorRT_Install_Directory>/data/ssd

## TensorRT Plugin layers in SSD

SSD has 3 plugin layers. They are Normalize, PriorBox and DetectionOutput. 
You can check ssd.prototxt and modify the plugin layer parameters similar 
to other caffe layers. The details about each layer and its parameters is 
shown below in caffe.proto format.

~~~~
message LayerParameter {
  optional DetectionOutputParameter detection_output_param = 881;
  optional NormalizeParameter norm_param = 882;
  optional PriorBoxParameter prior_box_param ==883;
}

// Message that stores parameters used by Normalize layer
NormalizeParameter {
  optional bool across_spatial = 1 [default = true];
  // Initial value of scale. Default is 1.0
  optional FillerParameter scale_filler = 2;
  // Whether or not scale parameters are shared across channels.
  optional bool channel_shared = 3 [default = true];
  // Epsilon for not dividing by zero while normalizing variance
  optional float eps = 4 [default = 1e-10];
}

// Message that stores parameters used by PriorBoxLayer
message PriorBoxParameter {
  // Encode/decode type.
  enum CodeType {
    CORNER = 1;
    CENTER_SIZE = 2;
    CORNER_SIZE = 3;
  }
  // Minimum box size (in pixels). Required!
  repeated float min_size = 1;
  // Maximum box size (in pixels). Required!
  repeated float max_size = 2;
  // Various aspect ratios. Duplicate ratios will be ignored.
  // If none is provided, we use default ratio 1.
  repeated float aspect_ratio = 3;
  // If true, will flip each aspect ratio.
  // For example, if there is aspect ratio "r",
  // we will generate aspect ratio "1.0/r" as well.
  optional bool flip = 4 [default = true];
  // If true, will clip the prior so that it is within [0, 1]
  optional bool clip = 5 [default = false];
  // Variance for adjusting the prior bboxes.
  repeated float variance = 6;
  // By default, we calculate img_height, img_width, step_x, step_y based on
  // bottom[0] (feat) and bottom[1] (img). Unless these values are explicitly
  // provided.
  // Explicitly provide the img_size.
  optional uint32 img_size = 7;
  // Either img_size or img_h/img_w should be specified; not both.
  optional uint32 img_h = 8;
  optional uint32 img_w = 9;

  // Explicitly provide the step size.
  optional float step = 10;
  // Either step or step_h/step_w should be specified; not both.
  optional float step_h = 11;
  optional float step_w = 12;

  // Offset to the top left corner of each cell.
  optional float offset = 13 [default = 0.5];
}

message NonMaximumSuppressionParameter {
  // Threshold to be used in NMS.
  optional float nms_threshold = 1 [default = 0.3];
  // Maximum number of results to be kept.
  optional int32 top_k = 2;
  // Parameter for adaptive NMS.
  optional float eta = 3 [default = 1.0];
}

// Message that stores parameters used by DetectionOutputLayer
message DetectionOutputParameter {
  // Number of classes to be predicted. Required!
  optional uint32 num_classes = 1;
  // If true, bounding box are shared among different classes.
  optional bool share_location = 2 [default = true];
  // Background label id. If there is no background class,
  // set it as -1.
  optional int32 background_label_id = 3 [default = 0];
  // Parameters used for NMS.
  optional NonMaximumSuppressionParameter nms_param = 4;

  // Type of coding method for bbox.
  optional PriorBoxParameter.CodeType code_type = 5 [default = CORNER];
  // If true, variance is encoded in target; otherwise we need to adjust the
  // predicted offset accordingly.
  optional bool variance_encoded_in_target = 6 [default = false];
  // Number of total bboxes to be kept per image after nms step.
  // -1 means keeping all bboxes after nms step.
  optional int32 keep_top_k = 7 [default = -1];
  // Only consider detections whose confidences are larger than a threshold.
  // If not provided, consider all boxes.
  optional float confidence_threshold = 8;
  // If true, visualize the detection results.
  optional bool visualize = 9 [default = false];
  // The threshold used to visualize the detection results.
}

~~~~

## Generate INT8 calibration batches

* Run `prepareINT8CalibrationBatches.sh` to generate INT8 bacthes. It select 500 
random JPEG images from PASCAL VOC dataset and convert to PPM images. These 500 
PPM images is used to generate INT8 calibration batches. 
* Please keep the batch files at <TensorRT_Install_Directory>/data/ssd/batches 
directory.
* If you want to use a different dataset to generate INT8 batches, please use 
batchPrepare.py and place the batch files in <TensorRT_Install_Directory>/data/ssd/batches directory.

## Usage

This sample can be run as:

    ./sample_ssd [-h] [--mode FP32/FP16/INT8]

