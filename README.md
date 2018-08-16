# Inatall AI envirnment on Ubuntu platform

   
**安装环境**  
*其他环境类似* 
```
硬件环境: CPU i7 / NVIDIA TITAN V
系统环境：Ubuntu 16.04 64bit
安装软件：CUDA9.0 / caffe / YOLO V3 / Protobuf / Matlab
````

### 目录
1. [参考](#1.-参考)
2. [安装Ubuntu和Windows双系统](#2.-安装ubuntu和windows双系统)   
3. [安装NVIDIA驱动](#3.-安装nvidia驱动)   
4. [安装CUDA 9.0](#4.-安装cuda-9.0)   
5. [安装cuDNN](#id4)  
6. [安装OpenCV](#id5) 
7. [安装Caffe 1.0](#id6)   
    7.1 Python2 下安装    
    7.2 Python3下安装    
8. [安装YOLO V3](#id7)
9.  [安装Protobuf](#id8)
10. [Linux MATLAB 2018a 安装教程及启动失败解决办法](#id9)
---
##  1. 参考  
1. https://blog.csdn.net/s717597589/article/details/79117112/
2. https://blog.csdn.net/balixiaxuetian/article/details/79154013
3. http://www.yaoingwen.com/ubuntu16-04-anaconda-3-6-caffe/
4. https://blog.csdn.net/m0_37407756/article/details/70789271
5. https://blog.csdn.net/ouening/article/details/79751393

---
## 2. 安装Ubuntu和Windows双系统  
详细的安装双系统就不过多介绍了，可以参考[这篇文章](https://blog.csdn.net/s717597589/article/details/79117112/)，但是在安装过程中有几个问题需要说明：      
- 安装Ubuntu之前首先要把BIOS的`security boot`关闭，否则会出现NVIDIA驱动安装完以后重启电脑会反复进入登录界面。
 
- 硬盘分区的时候可以只分为`swap`、`UEFI`、`\`和`\home`四个分区，不分`\home也可以，在挂在`\`分区的时候会自动生成`\home`和其他分区，但是在重装系统的时候`\home`无法重新挂在之前的`\home`分区导致数据丢失（类似于Windows的非系统盘）。
 
- 重装Ubuntu系统时请早Windows下用EasyUEFI软件将Ubuntu的引导项删除。

---
## 3. 安装NVIDIA驱动   

### 3.1. **在终端里依次输入以下命令安装驱动所需的依赖包**：   
```python
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler

sudo apt-get install --no-install-recommends libboost-all-dev
 
sudo apt-get install libopenblas-dev liblapack-dev libatlas-base-dev
 
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
 
sudo apt-get install git cmake build-essential
```
假如有安装包一直下载失败，可以使用：
```
sudo apt-get update 
```
### 3.2. **禁用Ubuntu自带的显卡驱动**：  
Ubuntu 16.04 自带 nouveau显卡驱动，这个自带的驱动是不能用于CUDA的，需要卸载重装。假如重装过显卡驱动则可跳过这一步。没有重装过的就跟着我的步骤往下走。

首先得禁用Ubuntu自带的显卡驱动nouveau，只有在禁用掉 nouveau 后才能顺利安装 NVIDIA 显卡驱动，禁用方法就是在 `/etc/modprobe.d/blacklist-nouveau.conf`文件中添加一条禁用命令，首先需要打开该文件，通过以下命令打开：   
```
sudo gedit /etc/modprobe.d/blacklist-nouveau.conf
```
打开后发现该文件中没有任何内容，**写入**：
```
blacklist nouveau  
options nouveau modeset=0
```
保存后关闭文件，注意此时还需执行以下命令使禁用 nouveau 真正生效：
```
sudo update-initramfs -u
```
**重启电脑**，然后输入以下命令，若什么都没有显示则禁用nouveau生效了：
```
lsmod | grep nouveau
```
### 3.3. **安装NVIDIA官方显卡驱动**：
通过`Ctrl + Alt + F1`进入文本模式，输入帐号密码登录，通过`Ctrl + Alt + F7`可返回图形化模式，在文本模式登录后首先关闭桌面服务：
```
sudo service lightdm stop
```
这里会要求你输入账户的密码。然后通过`Ctrl + Alt + F7`发现已无法成功返回图形化模式，说明桌面服务已成功关闭，注意此步对接下来的 nvidia 驱动安装尤为重要，必需确保桌面服务已关闭。按`Ctrl + Alt + F1`再次进入文本模式，先卸载之前的显卡驱动：    
```
sudo apt-get purge nvidia*
```
加入官方ppa源：   
```
sudo add-apt-repository ppa:graphics-drivers/ppa
```
之后刷新软件库并安装显卡驱动：   
```
sudo apt-get update
 
sudo apt-get install nvidia-390 nvidia-settings nvidia-prime  #大部分NVIDIA驱动可以安装390
```
**重启电脑**，通过下面命令查看显卡信息：   
···
nvidia-settings
···

### 3.4. **配置环境变量**：
使用 gedit 命令打开配置文件：
```
sudo gedit ~/.bashrc
```
打开后在**文件最后加入**以下两行内容：   
```
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
 
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```
保存并退出，运行以下内容使环境变量生效：   
```
source  ~./bashrc
```
---
## 4. 安装CUDA 9.0  
安装完显卡驱动后，CUDA toolkit和samples可单独安装，直接在终端运行安装，无需进入文本模式：   
```
sudo sh cuda_9.0.176_384.81_linux.run --no-opengl-libs
```
执行此命令约1分钟后会出现安装协议要你看，刚开始是0%，此时长按回车键让此百分比增长，直到100%，然后按照提示操作即可，先输入 accept ，是否安装显卡驱动选择no:   
```
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 387.26?
(y)es/(n)o/(q)uit: n
```
其余的一律按默认或者y进行安装即可。    
安装完成后配置CUDA环境变量，使用 gedit 命令打开配置文件：   
```
sudo gedit ~/.bashrc
```
在该文件最后加入以下两行并保存：   
```
export PATH=/usr/local/cuda/bin:$PATH    #/usr/local/cuda和/usr/local/cuda-9.0是同一个文件夹，两者通过软连接相连
 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
使该配置生效：   
```
source  ~/.bashrc
```
检验CUDA 是否安装成功，输入：   
```
cd /usr/local/cuda-9.1/samples/1_Utilities/deviceQuery
 
sudo make
 
./deviceQuery
```
卸载CUDA 9.1 的方法：   
``` 
cd /usr/local/cuda/bin
sudo ./uninstall_cuda_9.0.pl
```
卸载完成后如果显示：Not removing directory, it is not empty: /usr/local/cuda-9.0 ，假如需要重装CUDA 9.0的话就把这个文件夹删除。在/usr/local/路劲下输入：
```
sudo rm -r cuda-9.0
```
---
## 5. <span id="id4">安装cuDNN</span>   

解压cuNDD v7.zip到当前文件夹，得到一个cudn 文件夹，该文件夹下有include 和 lib64 两个文件夹，命令行进入其中的include 文件夹路径下，然后进行以下操作：
```
sudo cp cudnn.h /usr/local/cuda/include/ #复制头文件
```
然后命令行进入 cudn/lib64 文件夹路径下，运行以下命令：
```
sudo cp lib* /usr/local/cuda/lib64/ #复制动态链接库
cd /usr/local/cuda/lib64/ 
sudo rm -rf libcudnn.so libcudnn.so.7  #删除原有动态文件
sudo ln -s libcudnn.so.7.0.5 libcudnn.so.7  #生成软衔接
sudo ln -s libcudnn.so.7 libcudnn.so  #生成软链接
```
随后需要将路径/usr/local/cuda/lib64 添加到动态库，分两步：
1）安装vim。输入： 
```
sudo apt-get install vim-gtk
```
2）输入：
```
sudo vim /etc/ld.so.conf.d/cuda.conf
```
编辑状态下，输入：   
```
/usr/local/cuda/lib64
```
保存退出，输入下面代码使其生效：
```
sudo ldconfig
```
安装完成后可用`nvcc -V`命令验证是否安装成功，若出现以下信息则表示安装成功：
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Nov__3_21:07:56_CDT_2017
Cuda compilation tools, release 9.0, V9.0.85
```


---
## 6. <span id="id5">安装 OpenCV</span>   
进入官网 : http://opencv.org/releases.html , 选择 3.4.0 版本的 sources , 下载 opencv-3.4.0.zip 。随后解压到你要安装的位置，命令行进入已解压的文件夹 opencv-3.4.0 目录下，执行：
```
mkdir build # 创建编译的文件目录
 
cd build
 
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
 
make -j8  #编译
```
**遇到一下报错信息有两种可能：**
![编译报错](img/img1.png)
- 在编译opencv3.4.0源码的时候，会下载诸如ippicv_2017u3_lnx_intel64_20170822.tgz的东西，如果下载失败，请下载离线包（source文件夹中），解压该文件，会得到.cache文件夹，用此文件夹覆盖opencv源码文件夹下的.cache文件夹，再重新编译即可。.cahce文件夹为隐藏文件，可用ctrl+h查看。

- 若本机里安装了anaconda，则需要在环境变量(`sudo gedit ~/.bashrc`)中加入：
```
# added by Anaconda3 installer
export PATH="/home/andy/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=~/anaconda3/include/python3.6m
export PATH="$PATH:$HOME/bin"
```
在98%的时候会等很久很久，属于正常现象。编译过程很顺利，编译成功后安装：   
```
sudo make install #安装
```
安装完成后通过查看 opencv 版本验证是否安装成功：   
```
pkg-config --modversion opencv
```
接下来要给系统加入opencv库的环境变量:    
用gedit打开`/etc/ld.so.conf`，注意要用sudo打开获得权限，不然无法修改， 如：
```
sudo gedit /etc/ld.so.conf
```
在文件中加上一行:
```
/usr/local/lib
```
`/user/local`是opencv安装路径 就是makefile中指定的安装路径.

再运行`sudo ldconfig`, 修改`bash.bashrc`文件:
```
sudo gedit /etc/bash.bashrc
```
在文件末尾加入： 
```
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig 
export PKG_CONFIG_PATH 
```
运行`source /etc/bash.bashrc`使其生效。

卸载OpenCV的方法：进入OpenCV解压文件夹中的buid 文件夹：   
```
cd /home/ccem/opencv-3.4.0/build
```
运行：   
```
sudo make uninstall
```
然后把整个opencv-3.4.0文件夹都删掉。随后再运行：   
```
sudo rm -r /usr/local/include/opencv2 /usr/local/include/opencv /usr/include/opencv /usr/include/opencv2 /usr/local/share/opencv /usr/local/share/OpenCV /usr/share/opencv /usr/share/OpenCV /usr/local/bin/opencv* /usr/local/lib/libopencv
```
把一些残余的动态链接文件和空文件夹删掉。有些文件夹已经被删掉了所以会找不到路径。

---
## 7. <span id="id6">安装Caffe 1.0</span>    
### 7.1 Python2.7下安装caffe（推荐） <font color=red>需要Python2.7下安装OpenCV</font>
#### 7.1.1 装依赖库   
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y build-essential cmake git pkg-config
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev protobuf-compiler

sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y --no-install-recommends libboost-all-dev

sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get -y install build-essential cmake git libgtk2.0-dev pkg-config python-dev python-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip
```
#### 7.1.2 配置CUDA 及 CUDNN   
添加 CUDA 环境变量   
```
sudo gedit ~/.bashrc
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
```
Conda 安装依赖模块: 
```
conda install -c menpo opencv3
conda install libgcc
conda install protobuf
```
#### 7.1.3  <font color=red>需要Python2.7下安装OpenCV</font>
`方法同 6 `

#### 7.1.3  在caffe源码目录中新建`Makefile.config`文件，添加内容如下：
```
## Refer to http://caffe.berkeleyvision.org/installation.html
# Contributions simplifying and improving our build system are welcome!

# cuDNN acceleration switch (uncomment to build with cuDNN).
USE_CUDNN := 1

# CPU-only switch (uncomment to build without GPU support).
# CPU_ONLY := 1

# uncomment to disable IO dependencies and corresponding data layers
# USE_OPENCV := 0
# USE_LEVELDB := 0
# USE_LMDB := 0

# uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
#	You should not set this flag if you will be reading LMDBs with any
#	possibility of simultaneous read and write
# ALLOW_LMDB_NOLOCK := 1

# Uncomment if you're using OpenCV 3
OPENCV_VERSION := 3

# To customize your choice of compiler, uncomment and set the following.
# N.B. the default for Linux is g++ and the default for OSX is clang++
CUSTOM_CXX := g++

# CUDA directory contains bin/ and lib/ directories that we need.
CUDA_DIR := /usr/local/cuda
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 through *_61 lines for compatibility.
# For CUDA < 8.0, comment the *_60 and *_61 lines for compatibility.
# For CUDA >= 9.0, comment the *_20 and *_21 lines for compatibility.
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
BLAS := atlas
# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

# Homebrew puts openblas in a directory that is not on the standard search path
# BLAS_INCLUDE := $(shell brew --prefix openblas)/include
# BLAS_LIB := $(shell brew --prefix openblas)/lib

# This is required only if you will compile the matlab interface.
# MATLAB directory should contain the mex binary in /bin.
# MATLAB_DIR := /usr/local
# MATLAB_DIR := /Applications/MATLAB_R2012b.app

# NOTE: this is required only if you will compile the python interface.
# We need to be able to find Python.h and numpy/arrayobject.h.
PYTHON_INCLUDE := /usr/include/python2.7 \
		/usr/lib/python2.7/dist-packages/numpy/core/include
# Anaconda Python distribution is quite popular. Include path:
# Verify anaconda location, sometimes it's in root.
# ANACONDA_HOME := $(HOME)/anaconda
# PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		# $(ANACONDA_HOME)/include/python2.7 \
		# $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include

# Uncomment to use Python 3 (default is Python 2)
# PYTHON_LIBRARIES := boost_python3 python3.5m
# PYTHON_INCLUDE := /usr/include/python3.5m \
#                 /usr/lib/python3.5/dist-packages/numpy/core/include

# We need to be able to find libpythonX.X.so or .dylib.
PYTHON_LIB := /usr/lib
# PYTHON_LIB := $(ANACONDA_HOME)/lib

# Homebrew installs numpy in a non standard path (keg only)
# PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
# PYTHON_LIB += $(shell brew --prefix numpy)/lib

# Uncomment to support layers written in Python (will link against Python libs)
# WITH_PYTHON_LAYER := 1

# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib

# If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
# INCLUDE_DIRS += $(shell brew --prefix)/include
# LIBRARY_DIRS += $(shell brew --prefix)/lib

# NCCL acceleration switch (uncomment to build with NCCL)
# https://github.com/NVIDIA/nccl (last tested version: v1.2.3-1+cuda8.0)
# USE_NCCL := 1

# Uncomment to use `pkg-config` to specify OpenCV library paths.
# (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
# USE_PKG_CONFIG := 1

# N.B. both build and distribute dirs are cleared on `make clean`
BUILD_DIR := build
DISTRIBUTE_DIR := distribute

# Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
# DEBUG := 1

# The ID of the GPU that 'make runtest' will use to run unit tests.
TEST_GPUID := 0

# enable pretty build (comment to see full commands)
Q ?= @
```
#### 7.1.4 在caffe源码目录中修改`Makefile`文件中这一行如下：
```
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial
```

### 7.2 Python3下安装Caffe
#### 7.2.1 装依赖库   
`同Python2.7`
#### 7.2.2 配置CUDA 及 CUDNN   
`同Python2.7`da install protobuf

**conda install -c menpo opencv3命令有时候会显示权限不够permission issue。这是因为你安装anaconda时用的是sudo，这时候需要修改anaconda3文件夹权限**:
```
sudo chown -R 你的用户名（user ） /home/你的用户名/anaconda3
```
添加Anaconda CPLUS路径:   
```
export CPLUS_INCLUDE_PATH=~/anaconda3/include/python3.6m
```
配置 boost_python   
```
cd /usr/lib/x86_64-linux-gnu && sudo ln -s libboost_python-py35.so libboost_python3.so
```
在caffe源码目录中修改Makefile.config文件如下：
```
## Refer to http://caffe.berkeleyvision.org/installation.html
# Contributions simplifying and improving our build system are welcome!

# cuDNN acceleration switch (uncomment to build with cuDNN).
USE_CUDNN := 1

# Uncomment if you're using OpenCV 3
OPENCV_VERSION := 3

# CUDA directory contains bin/ and lib/ directories that we need.
CUDA_DIR := /usr/local/cuda
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them.
# For CUDA >= 9.0, comment the *_20 and *_21 lines for compatibility.
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
-gencode arch=compute_35,code=sm_35 \
-gencode arch=compute_50,code=sm_50 \
-gencode arch=compute_52,code=sm_52 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_61,code=sm_61 \
-gencode arch=compute_61,code=compute_61

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
BLAS := atlas

# NOTE: this is required only if you will compile the python interface.
# We need to be able to find Python.h and numpy/arrayobject.h.
# PYTHON_INCLUDE := /usr/include/python2.7 \
# /usr/lib/python2.7/dist-packages/numpy/core/include
# Anaconda Python distribution is quite popular. Include path:
# Verify anaconda location, sometimes it's in root.
ANACONDA_HOME := $(HOME)/anaconda3
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
# $(ANACONDA_HOME)/include/python3.6m \
# $(ANACONDA_HOME)/lib/python3.6/site-packages/numpy/core/include

# Uncomment to use Python 3 (default is Python 2)
PYTHON_LIBRARIES := boost_python3 python3.6m
# PYTHON_INCLUDE := /usr/include/python3.5m \
# /usr/lib/python3.5/dist-packages/numpy/core/include

# We need to be able to find libpythonX.X.so or .dylib.
#PYTHON_LIB := /usr/lib
PYTHON_LIB := $(ANACONDA_HOME)/lib

# Uncomment to support layers written in Python (will link against Python libs)
WITH_PYTHON_LAYER := 1

# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib

# If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
# INCLUDE_DIRS += $(shell brew --prefix)/include
# LIBRARY_DIRS += $(shell brew --prefix)/lib

# N.B. both build and distribute dirs are cleared on `make clean`
BUILD_DIR := build
DISTRIBUTE_DIR := distribute

# The ID of the GPU that 'make runtest' will use to run unit tests.
TEST_GPUID := 0

# enable pretty build (comment to see full commands)
Q ?= @
```

保存后,开始编译:   
```
make all -j $(($(nproc) + 1))
make test -j $(($(nproc) + 1))
make runtest -j $(($(nproc) + 1))
make pycaffe -j $(($(nproc) + 1))
```

<font color=red>在编译的时候有可能会提示错误，可能是因为anaconda环境的原因，需要在~/.bashrc中将anaconda的环境变量屏蔽掉,</font> **<font color=red>然后重启电脑</font>** 

添加环境变量
```
sudo gedit ~/.bashrc
export PYTHONPATH=~/caffe/python:$PYTHONPATH
```

### 7.3. <font color=red>caffe编译遇到的问题 </font>

<table><tr><td bgcolor=Violet>错误：protoc: error while loading shared libraries: libprotoc.so.10: cannot open shared object file: No such file or directory</td></tr></table>

**解决：**

```
export LD_LIBRARY_PATH=/usr/local/lib
```

<table><tr><td bgcolor=Violet>错误：/sbin/ldconfig.real: /usr/local/cuda-8.0/lib64/libcudnn.so.5 不是符号连接</td></tr></table>

**解决：**

在sudo ldconfig时遇到`usr/local/cuda-8.0/lib64/libcudnn.so.5 `不是符号连接的问题，解决办法也很简单，重新建立链接并删除原链接

首先找到`usr/local/cuda-8.0/lib64/`目录，搜索` libcudnn `然后发现两个文件`libcudnn.so.5`和`libcudnn.so.5.0.5 `理论上只有一个`libcudnn.so.5.0.5`

终端执行:
```
ln -sf /usr/local/cuda-8.0/lib64/libcudnn.so.5.0.5 /usr/local/cuda-8.0/lib64/libcudnn.so.5 
```
再`sudo ldconfig`时就可以了，这时候会发现usr/local/cuda-8.0/lib64/目录下只有`libcudnn.so.5.0.5`文件了，`libcudnn.so.5`消失了。

<table><tr><td bgcolor=Violet>错误：.build_release/tools/caffe: error while loading shared libraries: libhdf5.so.10: cannot open shared object file: No such file    or directory</td></tr></table>

**解决：**
```
echo "export LD_LIBRARY_PATH=/home/abc/anaconda2/lib:$LD_LIBRARY_PATH" >>~/.bashrc
```
<table><tr><td bgcolor=Violet>/usr/lib/x86_64-linux-gnu/libopencv_highgui.so：对‘TIFFOpen@LIBTIFF_4.0’未定义的引用
/usr/lib/x86_64-linux-gnu/libopencv_highgui.so：对‘TIFFReadEncodedStrip@LIBTIFF_4.0’未定义的引用
/usr/lib/x86_64-linux-gnu/libopencv_highgui.so：对‘TIFFSetField@LIBTIFF_4.0’未定义的引用
/usr/lib/x86_64-linux-gnu/libopencv_highgui.so：对‘TIFFSetWarningHandler@LIBTIFF_4.0’未定义的引用
/usr/lib/x86_64-linux-gnu/libopencv_highgui.so：对‘TIFFSetErrorHandler@LIBTIFF_4.0’未定义的引用
collect2:错误：ld返回1
make:*** [.build_release/tools/extract_features.bin]错误1</td></tr></table>

**解决：**

这个可能是权限问题，采用以下指令：
```
sudo su； 
make all ； 
make test ； 
make runtest ； 
make pycaffe； 
```
一切都能顺利解决

<table><tr><td bgcolor=Violet>错误：python/caffe/_caffe.cpp:1:52:致命错误：Python.h：没有那个文件或目录
编译中断。
make:***    
[python/caffe/_caffe.so]错误1</td></tr></table>

**解决：**

执行：`sudofind / -name 'Python.h'`找到他的路径，
在`Makefile.config`的PYTHON_INCLUDE加上`/home/abc/anaconda2/include/python2.7`（路径是自己的）

<table><tr><td bgcolor=Violet>错误：import caffe时：ImportError:No module named skimage.io</td></tr></table>

**解决：**
可能是我们没有安装所谓的skimage.io模块，所以可以用以下的命令来安装：
```
pip install scikit-image
```
关掉终端，重新进入再编译，或者：
```
sudo pip install scikit-image
```
另一种方法：
```
sudo apt-get install python-skimage
```
<table><tr><td bgcolor=Violet>错误：  

import caffe 

Traceback(most recent call last):   

File"<stdin>", line 1, in <module>     

ImportError:No module named caffe</td></tr></table>

**解决：**
```
echo'export PATH="/home/abc/caffe-master/python:$PATH"' >>~/.bashrc

source~/.bashrc
```
关掉终端，重新进入再编译





---
##  8. <span id="id8">安装 YOLO V3 </span>  

```
git clone http://github.com/pjreddie/darknet.git
cd darknet
```
根据YOLO[官方](https://pjreddie.com/darknet/install/)安装指导安装即可，其中若编译的时候用到opencv库的时候，遇到以下报错：
![编译报错](img/img2.png)
**原因是opencv没有加入到环境变量中，解决方式**
用gedit打开`/etc/ld.so.conf`，注意要用sudo打开获得权限，不然无法修改， 如：
```
sudo gedit /etc/ld.so.conf
```
在文件中加上一行:
```
/usr/local/lib
```
`/user/local`是opencv安装路径 就是makefile中指定的安装路径.

再运行`sudo ldconfig`, 修改`bash.bashrc`文件:
```
sudo gedit /etc/bash.bashrc
```
在文件末尾加入： 
```
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig 
export PKG_CONFIG_PATH 
```
运行`source /etc/bash.bashrc`使其生效。

---
##  9. <span id="#id8">安装Protobuf</span>
### protobuf是什么？
protobuf（Protocol Buffer）它是google提供的一个开源库，是一种语言无关、平台无关、扩展性好的用于通信协议、数据存储的结构化数据串行化方法。有如XML，不过它更小、更快、也更简单。你可以定义自己的数据结构，然后使用代码生成器生成的代码来读写这个数据结构。


### protobuf-c 是什么？
由于Protocol Buffer原生没有对C的支持，只能使用protobuf-c这个第三方库，它提供了支持C语言的API接口。

下面先安装protobuf，然后安装protobuf-c 。

### 9.1. 安装protocbuf
#### 9.1.1. 下载源码安装包
https://developers.google.com/protocol-buffers/
![下载界面](img/img3.png)
![GitHub界面](img/img4.png)
在release下可以找到所有的版本，我这里用的是2.4.1版本，复制protobuf-2.4.1.tar.gz的链接然后用wget命令下载。
```
wget https://github.com/google/protobuf/releases/download/v2.4.1/protobuf-2.4.1.tar.gz
```
#### 9.1.2. 解压
```
tar -zxvf protobuf-2.4.1.tar.gz
```
#### 9.1.3.  编译/安装
```
cd protobuf-2.4.1
```
（可以参考README思路来做。）
```
./configure
make
make check  #(check结果可能会有错误，但不用管她，因为暂时那些功能用不到)
make install
```
（完了之后会在 /usr/local/bin 目录下生成一个可执行文件 protoc）

#### 9.1.4. 检查安装是否成功
```
protoc --version
```
如果成功，则会输出版本号信息。如果有问题，则会输出错误内容。

#### 9.1.5.  错误及解决方法
```
protoc: error while loading shared libraries: libprotoc.so.8: cannot open shared
```
**错误原因**：
protobuf的默认安装路径是/usr/local/lib，而/usr/local/lib 不在Ubuntu体系默认的 LD_LIBRARY_PATH 里，所以就找不到该lib
解决方法：
1). 创建文件`sudo gedit /etc/ld.so.conf.d/libprotobuf.conf`，在该文件中输入如下内容：
```
/usr/local/lib
```
2). 执行命令
```
sudo ldconfig 
```
这时，再运行protoc --version 就可以正常看到版本号了


### 9.2. 安装protobuf-c
（这里使用的是protobuf-c-0.15版本，较高版本的安装类似）

进入下面的链接
https://code.google.com/p/protobuf-c/
进入Downloads界面
![下载界面](img/img5.png)
![下载界面](img/img6.png)
![下载界面](img/img7.png)

不知怎地，wget无法下载途中的`protobuf-c-0.15.tar.gz`文件。

怎么办呢，我们可以点击上图中的Export to GitHub，将代码导入到GitHub（当然你得有并登录自己的github账号），不过只有源码，没有release版。我们先wget下载源码，解包。由于是源码，所以没有configure文件，但是可以通过执行`autogen.sh`来生成configure文件，之后的操作就和安装protobuf类似了，这里就不细说了。
安装完成后会在` /usr/local/bin `目录下便会生成一个可执行文件 protoc-c

在安装完protobuf-c后，我们来检验一下protobuf-c是否安装成功。到 protobuf-c-0.15/src/test 目录下，执行如下命令：
```
protoc-c --c_out=. test.proto
```
（c_out 标志是用来指定编译后所生成文件的输出路径，这里c_out指定的是当前目录。）
如果在c_out指定目录下能够生成 test.pb-c.c 和 test.pb-c.h 这两个文件则说明安装成功了。

### 9.3. Protobuf的使用示例
```
touch person.proto
```
输入如下内容：
```
message Person {
  required string name = 1;
  required int32 id = 2;
}
```
编译.proto文件
```
protoc-c --c_out=. person.proto
```
```
touch main.c
```
输入如下代码：
```

#include <stdio.h>
#include <stdlib.h>
#include "person.pb-c.h"
 
void main()
{
        // 定义一个Person元素，并往其中存入数据
        Person person = PERSON__INIT;
        person.id = 1314;
        person.name = "lily";  // 字符串 lily 位于常量区
 
        printf("id = %d\n", person.id);
        printf("name = %s\n", person.name);
 
        // 打包
        int len = person__get_packed_size(&person);
        //printf("len = %d\n", len);
        void *sendpack = malloc(len);
        person__pack(&person, sendpack);
         // sendpack是打好的包，可以通过socket通讯将其发送出去。
        //（这里主要讲protobuf，就不发送了）
 
        // 接收端解包
        Person *recvbuf = person__unpack(NULL, len, sendpack);
        printf("id = %d\n", recvbuf->id);
        printf("name = %s\n", recvbuf->name);
        // 包用完了要释放
        person__free_unpacked(recvbuf, NULL);
        free(sendpack);
}
 
 ```
编译
```
gcc person.pb-c.c main.c -lprotobuf-c
 ```
执行` ./a.out`，输出结果如下：
id = 1314
name = lily
id = 1314
name = lily


---
##  10. <span id="id9">Linux MATLAB 2018a 安装教程及启动失败解决办法</span>   

matlab2018a 文件在下面吾爱破解给出：https://www.52pojie.cn/thread-713093-1-1.html

最好在百度网盘下载，文件太大容易挂掉，下载完成后有3个文件 
![下载界面](img/img8.png)

crack文件里面有密钥、许可证文件和需要替换的文件，和win版本是一样的。

（1）在安装包目录下打开Linux终端，执行下列命令：
```
sudo mkdir /mnt/matlab
sudo mount -o loop R2018a_glnxa64_dvd1.iso /mnt/matlab
cd /mnt/
sudo ./mnt/matlab/install
```
经过上面步骤就能看到安装界面了，默认安装路径在/usr/local/MATLAB/R2018a/ ，

（2）注意，上面只是挂载了第一个安装包，等安装到60%左右的时候会提示插入第二张CD，此时在刚才安装包目录下再次打开一个终端，执行
```
sudo mount -o loop R2018a_glnxa64_dvd2.iso /mnt/matlab
```
挂载第二张CD。 
（3）安装完成后，将crack里面的R2018a/bin 文件复制替换到安装目录下/usr/local/MATLAB/R2018a/ 
![下载界面](img/img9.png)
```
sudo cp -rvf R2018a/bin /usr/local/MATLAB/R2018a/
```
（4）接下来在/usr/local/MATLAB/R2018a/bin 目录下打开matlab
```
sudo ./matlab
```
指向许可证文件，激活，等下再次启动MATLAB，之后我自己的就出错了，转达下面部分讨论的内容。

全部安装完matlab2018a之后启动报错，将crash报给了MathWorks Support，很快回复了，按照里面提供的方法解决了，真是佩服MathWorks的服务（惭愧为了使用simulink使用db，平时数值计算还是用octave或者Python）

（1）首先贴出我的错误代码提示:
```
--------------------------------------------------------------------------------
       Segmentation violation detected at 五 3月 30 00:05:20 2018 +0800
--------------------------------------------------------------------------------
Configuration:
  Crash Decoding           : Disabled - No sandbox or build area path
  Crash Mode               : continue (default)
  Default Encoding         : UTF-8
  Deployed                 : false
  Desktop Environment      : XFCE
  GNU C Library            : 2.26 stable
  Java Version             : Java 1.8.0_144-b01 with Oracle Corporation Java HotSpot(TM) 64-Bit Server VM mixed mode
  MATLAB Architecture      : glnxa64
  MATLAB Entitlement ID    : 6257193
  MATLAB Root              : /usr/local/MATLAB/R2018a
  MATLAB Version           : 9.4.0.813654 (R2018a)
  Operating System         : "Manjaro Linux"
  Process ID               : 1155
  Processor ID             : x86 Family 6 Model 69 Stepping 1, GenuineIntel
  Session Key              : 44bbb319-da74-462f-b0e8-79d2dd5ab281
  Static TLS mitigation    : Disabled: Unnecessary 1
  Window System            : The X.Org Foundation (11906000), display :0.0

Fault Count: 1


Abnormal termination

Register State (from fault):
  RAX = 0000000000000000  RBX = 00007f3007ff7cf0
  RCX = 000000000097b3c0  RDX = 000000000097b3c0
  RSP = 00007f30836ed6b8  RBP = 0000000000000002
  RSI = 00007fff62861f48  RDI = 0000000000000002

   R8 = 00007f30287a2820   R9 = 000000000000002f
  R10 = 00007f302879c640  R11 = 0000000000000206
  R12 = 00007fff62861f48  R13 = 000000000097b3c0
  R14 = 00007f3007ff7cf8  R15 = 0000000000000000

  RIP = 000000000000b4c0  EFL = 0000000000010246

   CS = 0033   FS = 0000   GS = 0000

Stack Trace (from fault):
[  0] 0x000000000000b4c0                                   <unknown-module>+00000000
```
原因是
```
This error occurs when your computer cannot load a certain font display library through MATLAB.
```
官方给出的解决办法：

https://cn.mathworks.com/matlabcentral/answers/364727-why-does-matlab-crash-on-linux-fedora-26-with-a-segmentation-violation-r2017b-or-later

就是下面的方法
```
cd  /usr/local/MATLAB/R2018b   (or wherever you may have installed MATLAB)
cd bin/glnxa64
mkdir exclude
mv libfreetype* exclude/
```
当然，我遇到的情况是这样，网上还有一些说`linstdc.so`库和Linux系统自带的版本区别造成的，我也按照方法改成系统的了，但这个不是我遇到的问题解决办法，如果大家遇到了一些crash，发送报给给support也是个不错的选择。
