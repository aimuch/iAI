#include "trtUtil.h"

#include <cassert>
#include <functional>
#include <numeric>

namespace nmtSample
{
int inferTypeToBytes(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT: return sizeof(float); break;
    case nvinfer1::DataType::kHALF: return sizeof(int16_t); break;
    default: assert(0); break;
    }
};

int getVolume(nvinfer1::Dims dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}
}
