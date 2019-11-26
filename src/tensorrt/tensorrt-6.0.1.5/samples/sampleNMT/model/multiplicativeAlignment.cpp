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

#include "multiplicativeAlignment.h"

#include <cassert>
#include <sstream>

namespace nmtSample
{
MultiplicativeAlignment::MultiplicativeAlignment(ComponentWeights::ptr weights)
    : mWeights(weights)
{
    // please refer to chpt_to_bin.py for the details on the format
    assert(mWeights->mMetaData.size() >= 3);
    mKernelWeights.type = static_cast<nvinfer1::DataType>(mWeights->mMetaData[0]);
    assert(mKernelWeights.type == nvinfer1::DataType::kFLOAT);
    mInputChannelCount = mWeights->mMetaData[1];
    mOutputChannelCount = mWeights->mMetaData[2];

    mKernelWeights.values = (void*) (&mWeights->mWeights[0]);
    mKernelWeights.count = mInputChannelCount * mOutputChannelCount;
}

void MultiplicativeAlignment::addToModel(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor* attentionKeys,
    nvinfer1::ITensor* queryStates,
    nvinfer1::ITensor** alignmentScores)
{
    auto mmLayer = network->addMatrixMultiply(
        *queryStates,
        false,
        *attentionKeys,
        true);
    assert(mmLayer != nullptr);
    mmLayer->setName("Raw Alignment Scores MM (Queries x Keys) in multiplicative attention");
    *alignmentScores = mmLayer->getOutput(0);
    assert(*alignmentScores != nullptr);
}

void MultiplicativeAlignment::addAttentionKeys(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor* memoryStates,
    nvinfer1::ITensor** attentionKeys)
{
    nvinfer1::Dims weightDims{2, {mInputChannelCount, mOutputChannelCount}, {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kCHANNEL}};
    auto constLayer = network->addConstant(weightDims, mKernelWeights);
    assert(constLayer != nullptr);
    constLayer->setName("Matrix in multiplicative attention");
    auto weights = constLayer->getOutput(0);
    assert(weights != nullptr);

    auto mmLayer = network->addMatrixMultiply(
        *memoryStates,
        false,
        *weights,
        false);
    assert(mmLayer != nullptr);
    mmLayer->setName("Attention Keys MM in multiplicative attention");
    *attentionKeys = mmLayer->getOutput(0);
    assert(*attentionKeys != nullptr);
}

int MultiplicativeAlignment::getSourceStatesSize()
{
    return mInputChannelCount;
}

int MultiplicativeAlignment::getAttentionKeySize()
{
    return mOutputChannelCount;
}

std::string MultiplicativeAlignment::getInfo()
{
    std::stringstream ss;
    ss << "Multiplicative Alignment, source states size = " << mInputChannelCount << ", attention keys size = " << mOutputChannelCount;
    return ss.str();
}
} // namespace nmtSample
