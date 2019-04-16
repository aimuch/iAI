#ifndef SAMPLE_NMT_CUDA_ERROR_
#define SAMPLE_NMT_CUDA_ERROR_

#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif // SAMPLE_NMT_CUDA_ERROR_
