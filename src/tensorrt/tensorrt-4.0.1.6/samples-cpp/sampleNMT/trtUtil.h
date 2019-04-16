#ifndef SAMPLE_NMT_TRT_UTIL_
#define SAMPLE_NMT_TRT_UTIL_

#include "NvInfer.h"

namespace nmtSample
{
int inferTypeToBytes(nvinfer1::DataType t);

int getVolume(nvinfer1::Dims dims);
}

#endif // SAMPLE_NMT_TRT_UTIL_
