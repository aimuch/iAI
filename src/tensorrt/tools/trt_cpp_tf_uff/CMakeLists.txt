cmake_minimum_required(VERSION 2.8)
project(demo)

set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_VERBOSE_BUILD ON)

# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")

set(TENSORRT_ROOT /home/andy/TensorRT)


cuda_add_executable(demo common.cpp main.cpp)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
  find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()

target_include_directories(demo PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(demo ${TENSORRT_LIBRARY})
target_link_libraries(demo ${OpenCV_LIBS} )
