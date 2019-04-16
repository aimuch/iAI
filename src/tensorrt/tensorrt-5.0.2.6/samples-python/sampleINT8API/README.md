# NVIDIA TensorRT Sample "sampleINT8API"

The sampleINT8API sample demonstrates how to:
- Use nvinfer1::ITensor::setDynamicRange to set per tensor dynamic range
- Use nvinfer1::ILayer::setPrecison to set computation precision of a layer
- Use nvinfer1::ILayer::setOutputType to set output tensor data type of a layer
- Overall the sample showcase how to perform INT8 Inference without using INT8 Calibration
- Supports Image classification onnx models - resnet50, vgg19, mobilenet.
  Models can be obtained from here: https://github.com/onnx/models/tree/master/models/image_classification

## Usage

This sample can be run as:

    ./sample_int8_api [-h or --help]
    ./sample_int8_api [-h or --help] [-m modefile] [-s per_tensor_dynamic_range_file] [-i image_file] [-r reference_file] [-d path/to/data/dir] [--verbose] [--useDLACore=<id>]

sampleINT8API needs following files to build the network and run inference:

* `<network>.onnx`       - The model file which contains the network and trained weights
* `reference_labels.txt` - Labels reference file i.e. ground truth imagenet 1000 class mappings
* `per_tensor_dynamic_range.txt` - Custom per tensor dynamic range file or User can simply override them by iterating through network layers
* `image_to_infer.ppm`   - PPM Image to run inference with

By default, the sample expects these files to be in `data/samples/int8_api/`. The list of default directories can be changed by adding one or
more paths with `-d /new/path` as a command line argument.
