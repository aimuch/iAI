# PyTensorRT Plugins

Python wrapper for TensorRT Plugin Library

(versioning scheme right now is [TensorRT Target version].[Python Revision for this TensorRT version])

## Installation

### Dependencies  
Regardless of how you want to install the python package you will need the target TensorRT version installed and in your ```LD_LIBRARY_PATH```

The current supported version is TensorRT

Make sure both the ```libnvinfer.so``` and ```libcaffe_parser.so``` are available  

Install ```swig >= 3.0``` and any dependencies used for your custom layer implementation

### Python Dependencies
You need to install ```tensorrt``` for your target python version

### Building from Distribution

Binaries will be provided in wheel format.

To install, simply:

```sh
pip install [wheel file]
```

### Building from Source 

If you want to build from the source in this directory then clone the repo and run

Variables to change default behavior:
CUDNN_VERSION - specify version of CUDNN to use, defaults to 7
CUDA_VERSION - specify version of CUDA to use, defaults to 9.0
NVINFER_VERSION - specify version of libnvinfer to use, defaults to 4
CUDA_ROOT_DIR - Specify the root directory of the cuda installation, defaults to /usr/local/cuda
CUDN_ROOT_DIR - Specify the root directory of the cudnn installation, defaults to /usr/lib/x86_64-linux-gnu

```sh
python setup.py install
```

### Creating a package 

If you want to wrap a package

```sh
python setup.py bdist_wheel
```
## Usage 

The point of this package is to allow custom layers to be used in with the TensorRT Python API. 
Writing classes in python to implement custom layers with common libraries like CuDNN  
is difficult so the proposed usage for this package is for users to 
implement their layers in C++ then complie into a python package using ```setup_tools```

After implementing your own layers in a similar manner to the Fully Connected Layer in ```/src```, 
add references to the header file in the two marked locations in ```/interfaces/plugin.i```
and add the source files to the sources list in ```setup.py``` if needed.


After building with your new plugin, you should be able to access the layer in 
```tensorrtplugins.[NAME OF YOUR PLUGIN]```
