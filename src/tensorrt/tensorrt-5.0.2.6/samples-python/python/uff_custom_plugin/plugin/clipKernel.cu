#include <clipKernel.h>

template <typename T>
__device__ __forceinline__ const T& min(const T& a, const T& b)
{
    return (a > b) ? b : a;
}

template <typename T>
__device__ __forceinline__ const T& max(const T& a, const T& b)
{
    return (a > b) ? a : b;
}

template <typename T, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void clipKernel(
        int n,
        const T clipMin,
        const T clipMax,
        const T* input,
        T* output)
{
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        output[i] = min<T>(max<T>(input[i], clipMin), clipMax);
    }
}

int clipInference(
    cudaStream_t stream,
    int n,
    float clipMin,
    float clipMax,
    const void* input,
    void* output)
{
    const int blockSize = 512;
    const int gridSize = (n + blockSize - 1) / blockSize;
    clipKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(n, clipMin, clipMax,
                                                 static_cast<const float*>(input),
                                                 static_cast<float*>(output));
    return 0;
}
