# About This Sample
This sample demonstrates how to use TensorRT and its including suite of parsers to perform inference with ResNet50 models trained with various different frameworks.   

# Parser Overview
TensorRT uses a suite of parsers to generate TensorRT networks from models trained in different frameworks.

## UFF Parser
The UFF parser is used for TensorFlow models. After freezing a TensorFlow graph and writing it to a protobuf file, you can convert it to UFF with the `convert-to-uff` utility included with TensorRT. This sample ships with a pre-generated UFF file.

## Caffe Parser
The Caffe parser is used for Caffe2 models. After training, you can invoke the caffe parser directly on the model file (usually `.caffemodel`) and deploy file (usually `.prototxt`).

## ONNX Parser
The ONNX parser can be used with any framework that supports the ONNX format. It can be used with `.onnx` files.

# Installing Prerequisites
1. Make sure you have the python dependencies installed.
    - For python2, run `python2 -m pip install -r requirements.txt` from the top-level of this sample.
    - For python3, run `python3 -m pip install -r requirements.txt` from the top-level of this sample.

# Running the Samples
1. Create a TensorRT inference engine and run inference:
    ```
 	python uff_resnet50.py [-d DATA_DIR]
	python caffe_resnet50.py [-d DATA_DIR]
	python onnx_resnet50.py [-d DATA_DIR]
    ```
    The data directory needs to be specified only if TensorRT is not installed in the default location.
