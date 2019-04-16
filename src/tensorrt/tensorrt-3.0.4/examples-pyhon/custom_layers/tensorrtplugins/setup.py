#
# Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

import os
import sys

try:
    from setuptools.command.build_ext import build_ext
    from setuptools import setup, Extension, Command, find_packages
except:
    from distutils.command.build_ext import build_ext
    from distutils import setup, Extension, Command, find_packages

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

PY_VERSION = sys.version_info

CUDNN_VERSION = os.environ.get("CUDNN_VERSION",'7')
CUDA_VERSION = os.environ.get("CUDA_VERSION", '9.0')
CUDA_DIR = os.environ.get("CUDA_ROOT_DIR", '/usr/local/cuda')
CUDNN_DIR = os.environ.get("CUDNN_ROOT_DIR", '/usr/lib/x86_64-linux-gnu')
TENSORRT_INC_DIR = '/usr/include/x86_64-linux-gnu'
TENSORRT_LIB_DIR = '/usr/lib/x86_64-linux-gnu'
CUDA_LIB_DIR = str(CUDA_DIR) + '/lib64'
CUDNN_LIB = str(CUDNN_DIR) + '/libcudnn.so.' + str(CUDNN_VERSION)
CUDNN_INC_DIR = str(CUDA_DIR) + '/include'
TENSORRT_VERSION = os.environ.get("NVINFER_VERSION",'4')

REQUIRED_PACKAGES = [
  'tensorrt >= 3.0.4'
]

SWIG_OPTS = ['-c++',
             '-v',
             '-modern',
             '-builtin',
             '-Wall',
             '-fvirtual',
             '-I' + TENSORRT_INC_DIR,
             '-I/usr/local/include',
             '-I/usr/include',
             '-I/usr/include/c++/4.8/']

if PY_VERSION.major > 2:
  SWIG_OPTS.append('-py3')

PYTHON_INC = '/usr/include/python' + str(PY_VERSION.major) + '.' + str(PY_VERSION.minor)
NUMPY_INC = '/usr/local/lib/python' + str(PY_VERSION.major) + '.' + str(PY_VERSION.minor) + "/numpy/core/include"

INC_DIRS = ['.',
            CUDNN_INC_DIR,
            '/usr/include/x86_64-linux-gnu',
            '/usr/local/cuda/include',
            '/usr/local/include',
             TENSORRT_INC_DIR,
             PYTHON_INC,
             '.',
             '/usr/include/x86_64-linux-gnu/',
             NUMPY_INC]

LIB_DIRS = [CUDA_LIB_DIR,
            CUDNN_DIR,
            TENSORRT_LIB_DIR]

LIBS = ['cudart',
        'nvinfer',
        'nvcaffe_parser',
        'cublas',
        'cudnn']

EXTRA_OBJS = [CUDA_LIB_DIR + '/libcudart.so',
              CUDNN_LIB,
              CUDA_LIB_DIR + '/libcublas.so',
              TENSORRT_LIB_DIR + '/libnvinfer.so.' + TENSORRT_VERSION,
              TENSORRT_LIB_DIR + '/libnvcaffe_parser.so.' + TENSORRT_VERSION,
              TENSORRT_LIB_DIR + '/libnvinfer_plugin.so.' + TENSORRT_VERSION]

EXTRA_COMPILE_ARGS =  ['-std=c++11',
                       '-DNDEBUG',
                       '-DUNIX',
                       '-D__UNIX',
                       '-m64',
                       '-fPIC',
                       '-O2',
                       '-w',
                       '-fmessage-length=0']

EXTRA_LINK_ARGS = ['-Wl,--no-as-needed']

#Wrap Plugins
plugin_module = Extension('tensorrtplugins._plugins',
                        sources=['interfaces/plugins.i'],
                        swig_opts = SWIG_OPTS,
                        library_dirs = LIB_DIRS,
                        libraries = LIBS,
                        extra_objects = EXTRA_OBJS,
                        include_dirs = INC_DIRS,
                        extra_compile_args = EXTRA_COMPILE_ARGS,
                        extra_link_args = EXTRA_LINK_ARGS)

setup (
    name = "tensorrtplugins",
    author = 'NVIDIA Corporation', 
    author_email = 'kismats@nvidia.com', 
    use_2to3 = True,
    version = "0.0.1",
    packages = find_packages(),
    ext_modules = [plugin_module],
    install_requires = REQUIRED_PACKAGES
)
