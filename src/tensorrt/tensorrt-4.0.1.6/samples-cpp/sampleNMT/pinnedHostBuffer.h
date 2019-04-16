#ifndef SAMPLE_NMT_PINNED_HOST_BUFFER_
#define SAMPLE_NMT_PINNED_HOST_BUFFER_

#include "cudaError.h"
#include <cuda_runtime_api.h>
#include <memory>

namespace nmtSample
{
/** \class PinnedHostBuffer
    *
    * \brief wrapper for the pinned host memory region  
    *
    */
template <typename T>
class PinnedHostBuffer
{
public:
    typedef std::shared_ptr<PinnedHostBuffer<T>> ptr;

    PinnedHostBuffer(size_t elementCount)
        : mBuffer(nullptr)
    {
        CUDA_CHECK(cudaHostAlloc(&mBuffer, elementCount * sizeof(T), cudaHostAllocDefault));
    }

    virtual ~PinnedHostBuffer()
    {
        if (mBuffer)
        {
            cudaFreeHost(mBuffer);
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

#endif // SAMPLE_NMT_PINNED_HOST_BUFFER_
