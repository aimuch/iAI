#ifndef SAMPLE_NMT_DEVICE_BUFFER_
#define SAMPLE_NMT_DEVICE_BUFFER_

#include "cudaError.h"
#include <cuda_runtime_api.h>
#include <memory>

namespace nmtSample
{
template <typename T>
class DeviceBuffer
{
public:
    typedef std::shared_ptr<DeviceBuffer<T>> ptr;

    DeviceBuffer(size_t elementCount)
        : mBuffer(nullptr)
    {
        CUDA_CHECK(cudaMalloc(&mBuffer, elementCount * sizeof(T)));
    }

    virtual ~DeviceBuffer()
    {
        if (mBuffer)
        {
            cudaFree(mBuffer);
        }
    }

    operator T*()
    {
        return mBuffer;
    }

    operator const T*() const
    {
        return mBuffer;
    }

protected:
    T* mBuffer;
};
}

#endif // SAMPLE_NMT_DEVICE_BUFFER_
