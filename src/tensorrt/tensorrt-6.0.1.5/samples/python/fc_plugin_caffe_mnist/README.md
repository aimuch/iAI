# Adding A Custom Layer To Your Caffe Network In TensorRT In Python


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, fc_plugin_caffe_mnist, demonstrates how to implement a custom FullyConnected layer using cuBLAS and cuDNN, wraps the implementation in a TensorRT plugin (with a corresponding plugin factory), and generates Python bindings for it using pybind11. These bindings are then used to register the plugin factory with the CaffeParser.

**Note:**  The Caffe InnerProduct/FullyConnected layer is normally handled natively in TensorRT using the IFullyConnected layer. However, in this sample, we use a plugin implementation for instructive purposes.

## How does this sample work?

This sample demonstrates how to use plugins written in C++ with the TensorRT Python bindings and the Caffe parser. This sample includes:

`plugin/`
This directory contains files for the FullyConnected layer plugin.

`FullyConnected.h`
This plugin implements CUDA, cuDNN, and cuBLAS.

`pyFullyConnected.cpp`
This plugin generates Python bindings for the `FCPlugin` and `FCPluginFactory` classes.

`sample.py`
This script runs an [MNIST network](http://yann.lecun.com/exdb/lenet/) using the provided FullyConnected layer plugin.

`requirements.txt`
This file specifies all the Python packages required to run this Python sample.

## Prerequisites

1. [Install cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).

2. [Install CMake](https://cmake.org/download/).

3. [Install cuBLAS](https://developer.nvidia.com/cublas).

4. Download `pybind11`.
    `git clone -b v2.2.3 https://github.com/pybind/pybind11.git`

	You can clone the repository anywhere, but the default configuration assumes that `pybind11` is located in your home directory.

5. Install the dependencies for Python.
	-   For Python 2 users, from the root directory, run:
		`python2 -m pip install -r requirements.txt`

	-   For Python 3 users, from the root directory, run:
		`python3 -m pip install -r requirements.txt`

## Running the sample

1.  Build the plugin and its corresponding Python bindings.
	```
	mkdir build && pushd build
	cmake .. -DCUDA_VERSION=10.1
	```
	where `CUDA_VERSION` is set to your local CUDA version.

	**Note:** If any of the dependencies are not installed in their default locations, you can manually specify them. For example:
	```
	cmake .. -DPYBIND11_DIR=/usr/local/pybind11/
	-DCUDA_ROOT=/usr/local/cuda-9.2/
	-DPYTHON3_INC_DIR=/usr/include/python3.6/
	-DNVINFER_LIB=/path/to/libnvinfer.so -DTRT_INC_DIR=/path/to/tensorrt/include/
	```

	`cmake ..` displays a complete list of configurable variables. If a variable is set to `VARIABLE_NAME-NOTFOUND`, then you’ll need to specify it manually or set the variable it is derived from correctly.

	The default behavior is to build bindings for both Python 2 and 3. To disable either one, for example, issue:
	`cmake .. -DPYTHON3_INC_DIR=None`

	to disable Python 3.

2. Build the plugin.
	```
	make -j4
	popd
	```

3. Run the sample to perform inference using the plugin. For example, issue:
   `python3 sample.py [-d DATA_DIR]`

	to run the sample with Python 3.

	**Note:** If the TensorRT sample data is not installed in the default location, for example `/usr/src/tensorrt/data/`, the data directory must be specified. For example:
	`python sample.py -d /path/to/my/data/`

	A single artifact called mnist.engine is created in the source directory and contains a serialized engine.

4. Verify that the sample ran successfully. If the sample runs successfully you should see a match between the test case and the prediction.
    ```
	Test Case: #
	Prediction: #
	```

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.

# Additional resources

The following resources provide a deeper understanding about adding a custom layer to your Caffe network using Python:

**Network**
- [MNIST network](http://yann.lecun.com/exdb/lenet/)

**Dataset**
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

February 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.
