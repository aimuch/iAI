/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include "debugUtil.h"

#include <cassert>
#include <cuda_runtime_api.h>

#include "../cudaError.h"

namespace nmtSample
{
std::list<DebugUtil::DumpTensorPlugin::ptr> DebugUtil::mPlugins;

DebugUtil::DumpTensorPlugin::DumpTensorPlugin(std::shared_ptr<std::ostream> out)
    : mOut(out)
{
}

int DebugUtil::DumpTensorPlugin::getNbOutputs() const
{
    return 1;
}

nvinfer1::Dims DebugUtil::DumpTensorPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    return inputs[0];
}

void DebugUtil::DumpTensorPlugin::configure(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, int maxBatchSize)
{
    mDims = inputDims[0];

    *mOut << "Max batch size = " << maxBatchSize << std::endl;
    *mOut << "Tensor dimensions = ";
    mTensorVolume = 1;
    for (int i = 0; i < mDims.nbDims; ++i)
    {
        if (i > 0)
            *mOut << "x";
        *mOut << mDims.d[i];
        mTensorVolume *= mDims.d[i];
    }
    mElemsPerRow = 1;
    for (int i = mDims.nbDims - 1; i >= 0; --i)
    {
        if (mElemsPerRow == 1)
            mElemsPerRow *= mDims.d[i];
    }
    *mOut << std::endl;

    mData = std::make_shared<PinnedHostBuffer<float>>(mTensorVolume * maxBatchSize);
}

int DebugUtil::DumpTensorPlugin::initialize()
{
    return 0;
}

void DebugUtil::DumpTensorPlugin::terminate()
{
    mOut.reset();
    mData.reset();
}

size_t DebugUtil::DumpTensorPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int DebugUtil::DumpTensorPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    int totalElems = batchSize * mTensorVolume;

    CUDA_CHECK(cudaMemcpyAsync(*mData, inputs[0], totalElems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], totalElems * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    *mOut << "Batch size = " << batchSize << "\n";
    int rowCount = totalElems / mElemsPerRow;
    for (int rowId = 0; rowId < rowCount; ++rowId)
    {
        for (int i = 0; i < mElemsPerRow; ++i)
        {
            if (i > 0)
                *mOut << " ";
            *mOut << (*mData)[rowId * mElemsPerRow + i];
        }
        *mOut << "\n";
    }
    *mOut << std::endl;

    return 0;
}

size_t DebugUtil::DumpTensorPlugin::getSerializationSize()
{
    assert(0);
    return 0;
}

void DebugUtil::DumpTensorPlugin::serialize(void* buffer)
{
    assert(0);
}

void DebugUtil::addDumpTensorToStream(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor* input,
    nvinfer1::ITensor** output,
    std::shared_ptr<std::ostream> out)
{
    assert(!input->getBroadcastAcrossBatch());
    auto plugin = std::make_shared<DumpTensorPlugin>(out);
    nvinfer1::ITensor* inputTensors[] = {input};
    auto pluginLayer = network->addPlugin(inputTensors, 1, *plugin);
    assert(pluginLayer != nullptr);
    *output = pluginLayer->getOutput(0);
    assert(*output != nullptr);
    mPlugins.push_back(plugin);
}
} // namespace nmtSample
