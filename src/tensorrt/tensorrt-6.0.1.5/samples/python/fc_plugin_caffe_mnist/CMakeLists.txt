cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(FCPlugin LANGUAGES CXX C)

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
# Set module name here. MUST MATCH the module name specified in the .cpp
set_ifndef(PY_MODULE_NAME fcplugin)
# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD 11)
# pybind11 defaults to c++14.
set(PYBIND11_CPP_STANDARD -std=c++11)

set(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)

set_ifndef(PYBIND11_DIR $ENV{HOME}/pybind11/)
set_ifndef(CUDA_VERSION 10.0)
set_ifndef(CUDA_ROOT /usr/local/cuda-${CUDA_VERSION})
set_ifndef(CUDNN_ROOT ${CUDA_ROOT})
set_ifndef(PYTHON_ROOT /usr/include)
set_ifndef(TRT_LIB_DIR /usr/lib/x86_64-linux-gnu)
set_ifndef(TRT_INC_DIR /usr/include/x86_64-linux-gnu)

# Find dependencies.
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

find_path(_CUDA_INC_DIR cuda_runtime_api.h HINTS ${CUDA_ROOT} PATH_SUFFIXES include)
set_ifndef(CUDA_INC_DIR ${_CUDA_INC_DIR})

find_library(_CUDA_LIB cudart HINTS ${CUDA_ROOT} PATH_SUFFIXES lib lib64)
set_ifndef(CUDA_LIB ${_CUDA_LIB})

find_library(_CUBLAS_LIB cublas HINTS ${CUDA_ROOT} PATH_SUFFIXES lib lib64)
set_ifndef(CUBLAS_LIB ${_CUBLAS_LIB})

find_path(_CUDNN_INC_DIR cudnn.h HINTS ${CUDNN_ROOT} PATH_SUFFIXES include x86_64-linux-gnu)
set_ifndef(CUDNN_INC_DIR ${_CUDNN_INC_DIR})

find_library(_CUDNN_LIB cudnn HINTS ${CUDNN_ROOT} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(CUDNN_LIB ${_CUDNN_LIB})

find_library(_TRT_INC_DIR NvInfer.h HINTS ${TRT_INC_DIR} PATH_SUFFIXES include x86_64-linux-gnu)
set_ifndef(TRT_INC_DIR ${_TRT_INC_DIR})

find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_library(_NVPARSERS_LIB nvparsers HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVPARSERS_LIB ${_NVPARSERS_LIB})

find_library(_NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVINFER_PLUGIN_LIB ${_NVINFER_PLUGIN_LIB})

find_path(_PYTHON2_INC_DIR Python.h HINTS ${PYTHON_ROOT} PATH_SUFFIXES python2.7)
set_ifndef(PYTHON2_INC_DIR ${_PYTHON2_INC_DIR})

find_path(_PYTHON3_INC_DIR Python.h HINTS ${PYTHON_ROOT} PATH_SUFFIXES python3.7 python3.6 python3.5 python3.4)
set_ifndef(PYTHON3_INC_DIR ${_PYTHON3_INC_DIR})

# -------- BUILDING --------

# Add include directories
include_directories(${TRT_INC_DIR} ${CUDA_INC_DIR} ${CUDNN_INC_DIR} ${PYBIND11_DIR}/include/)

# Add this so we can retrieve pybind11_add_module.
add_subdirectory(${PYBIND11_DIR} ${CMAKE_BINARY_DIR}/pybind11)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/plugin/*.cpp)

# Bindings library. The module name MUST MATCH the module name specified in the .cpp
if(PYTHON3_INC_DIR AND NOT (${PYTHON3_INC_DIR} STREQUAL "None"))
    pybind11_add_module(${PY_MODULE_NAME} SHARED THIN_LTO ${SOURCE_FILES})
    target_include_directories(${PY_MODULE_NAME} BEFORE PUBLIC ${PYTHON3_INC_DIR})
    target_link_libraries(${PY_MODULE_NAME} PRIVATE ${CUDNN_LIB} ${CUDA_LIB} ${CUBLAS_LIB} ${NVINFER_LIB} ${NVPARSERS_LIB} ${NVINFER_PLUGIN_LIB})
endif()

if(PYTHON2_INC_DIR AND NOT (${PYTHON2_INC_DIR} STREQUAL "None"))
    # Suffix the cmake target name with a 2 to differentiate from the Python 3 bindings target.
    pybind11_add_module(${PY_MODULE_NAME}2 SHARED THIN_LTO ${SOURCE_FILES})
    target_include_directories(${PY_MODULE_NAME}2 BEFORE PUBLIC ${PYTHON2_INC_DIR})
    target_link_libraries(${PY_MODULE_NAME}2 PRIVATE ${CUDNN_LIB} ${CUDA_LIB} ${CUBLAS_LIB} ${NVINFER_LIB} ${NVPARSERS_LIB} ${NVINFER_PLUGIN_LIB})
    # Rename to remove the .cpython-35... extension.
    set_target_properties(${PY_MODULE_NAME}2 PROPERTIES OUTPUT_NAME ${PY_MODULE_NAME} SUFFIX ".so")
    # Python 2 requires an empty __init__ file to be able to import.
    file(WRITE ${CMAKE_BINARY_DIR}/__init__.py "")
endif()
