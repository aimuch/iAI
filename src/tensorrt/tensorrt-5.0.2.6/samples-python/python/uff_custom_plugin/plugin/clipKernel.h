#ifndef CLIP_KERNEL_H
#define CLIP_KERNEL_H
#include "NvInfer.h"

int clipInference(
    cudaStream_t stream,
    int n,
    float clipMin,
    float clipMax,
    const void* input,
    void* output);

#endif
