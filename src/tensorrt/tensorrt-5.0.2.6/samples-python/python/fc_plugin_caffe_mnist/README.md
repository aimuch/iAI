# About This Sample
This sample demonstrates how to use plugins written in C++ with the TensorRT Python bindings and Caffe Parser. More specifically, this sample implements a fully connected layer using cuBLAS and cuDNN, wraps the implementation in a TensorRT plugin (with a corresponding plugin factory) and then generates python bindings for it using pybind11. These bindings are then used to register the plugin factory with the Caffe Parser.

# Installing Prerequisites
1. Install CUDA (https://developer.nvidia.com/cuda-toolkit)
2. Install cuDNN (https://developer.nvidia.com/cudnn)
3. Install cuBLAS (https://developer.nvidia.com/cublas)
4. Download pybind11 with `git clone -b v2.2.3 https://github.com/pybind/pybind11.git`. You can do this anywhere, but the default configuration assumes that pybind11 is located in your home directory.
5. Make sure you have the python dependencies installed.
    - For python2, run `python2 -m pip install -r requirements.txt` from the top-level of this sample.
    - For python3, run `python3 -m pip install -r requirements.txt` from the top-level of this sample.

# Sample Structure
- **plugin/** contains files for the FullyConnected layer plugin
    - **FullyConnected.h** implements the plugin using CUDA, cuDNN, and cuBLAS.
    - **pyFullyConnected.cpp** generates python bindings for the FCPlugin and FCPluginFactory classes.
- **sample.py** runs an MNIST network using the provided FullyConnected layer plugin.
- **requirements.txt** specifies all the python packages required to run the python sample.

# Building the Plugin
To build the plugin and its corresponding python bindings, run:
1. `mkdir build && pushd build`
2. `cmake ..`

    Note that if any of the dependencies are not installed in their default locations, you can manually specify them to cmake.
    For example, `cmake .. -DPYBIND11_DIR=/usr/local/pybind11/ -DCUDA_ROOT=/usr/local/cuda-9.2/ -DPYTHON3_INC_DIR=/usr/include/python3.6/ -DNVINFER_LIB=/path/to/libnvinfer.so -DTRT_INC_DIR=/path/to/tensorrt/include/`.

    `cmake ..` will display a complete list of configurable variables - if a variable is set to `VARIABLE_NAME-NOTFOUND`, then you probably need to specify it manually (or set the variable it is derived from correctly).

    The default behavior is to build bindings for both Python 2 and 3. To disable either one, you can run `cmake ..` with `PYTHON2_INC_DIR` or `PYTHON3_INC_DIR` set to `None`. For example, `cmake .. -DPYTHON3_INC_DIR=None` will disable Python 3 bindings.

3. `make -j4`
4. `popd`

# Running the Sample
Finally, you can execute the sample.
    - For python2 run `python2 sample.py [-d DATA_DIR]`
    - For python3 run `python3 sample.py [-d DATA_DIR]`
    The data directory needs to be specified only if TensorRT is not installed in the default location.
