# About This Sample
This sample demonstrates how to use plugins written in C++ with the TensorRT
Python bindings and UFF Parser. More specifically, this sample implements
a clip layer (as a CUDA kernel), wraps the implementation in a TensorRT plugin
(with a corresponding plugin creator) and then generates shared library module
containing its code. The user then dynamically links this library in Python,
which causes plugin to be registered in TensorRT's PluginRegistry and
makes it available for UFF parser.

# Installing Prerequisites
1. Install cmake >= 3.8 (https://cmake.org/cmake/help/latest/command/install.html)
2. Install CUDA (https://developer.nvidia.com/cuda-toolkit)
3. Make sure you have the python dependencies installed.
    - For python2, run `python2 -m pip install -r requirements.txt` from the top-level of this sample.
    - For python3, run `python3 -m pip install -r requirements.txt` from the top-level of this sample.
4. Make sure that you have tensorrt, graphsurgeon and uff installed

# Sample Structure
- **plugin/** contains files for the Clip layer plugin
    - **clipKernel.cu** CUDA kernel that clips input
    - **clipKernel.h** header exposing CUDA kernel to C++ code
    - **customClipPlugin.cpp** implementation of clip TensorRT plugin, which
    uses CUDA kernel internally
    - **customClipPlugin.h** ClipPlugin headers
- **lenet5.py** trains an MNIST network that uses ReLU6 activation
(not natively supported in TensorRT)
- **mnist_uff_relu6_plugin.py** transforms trained model into UFF (delegating
ReLU6 activation to ClipPlugin instance) and runs inference in TensorRT
- **requirements.txt** specifies all the Python packages required to run the
Python sample

# Building the Plugin
To build the plugin and its corresponding python bindings, run:
1. `mkdir build && pushd build`
2. `cmake ..`

    Note that if any of the dependencies are not installed in their default
    locations, you can manually specify them to cmake.

    For example:
    `cmake .. -DTRT_LIB=/home/dev/tensorrt -DTRT_INCLUDE=/home/dev/tensorrt/include -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`

    `cmake ..` will display a complete list of configurable variables:
    if a variable is set to `VARIABLE_NAME-NOTFOUND`, then you probably
    need to specify it manually (or set the variable it is derived from
    correctly).

3. `make -j8`
4. `popd`

# Running the Sample

Run MNIST training
- For python2 run `python2 lenet5.py`.
- For python3 run `python3 lenet5.py`.

Execute the sample.
- For python2 run `python2 mnist_uff_custom_plugin.py`.
- For python3 run `python3 mnist_uff_custom_plugin.py`.
