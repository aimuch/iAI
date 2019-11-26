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

#include "lstmDecoder.h"

#include "trtUtil.h"

#include "debugUtil.h"
#include <fstream>

#include <cassert>
#include <sstream>

namespace nmtSample
{
LSTMDecoder::LSTMDecoder(ComponentWeights::ptr weights)
    : mWeights(weights)
{
    // please refer to chpt_to_bin.py for the details on the format
    assert(mWeights->mMetaData.size() >= 4);
    nvinfer1::DataType dataType = static_cast<nvinfer1::DataType>(mWeights->mMetaData[0]);
    assert(dataType == nvinfer1::DataType::kFLOAT);
    mRNNKind = mWeights->mMetaData[1];
    mNumLayers = mWeights->mMetaData[2];
    mNumUnits = mWeights->mMetaData[3];
    size_t elementSize = inferTypeToBytes(dataType);
    // compute weights offsets
    size_t dataSize = 2 * mNumUnits;
    size_t kernelOffset = 0;
    size_t biasStartOffset = ((4 * dataSize + 4 * mNumUnits) * mNumUnits) * elementSize
        + 8 * mNumUnits * mNumUnits * (mNumLayers - 1) * elementSize;
    size_t biasOffset = biasStartOffset;
    int numGates = 8;
    for (int layerIndex = 0; layerIndex < mNumLayers; layerIndex++)
    {
        for (int gateIndex = 0; gateIndex < numGates; gateIndex++)
        {
            // encoder input size == mNumUnits
            int64_t inputSize = ((layerIndex == 0) && (gateIndex < 4)) ? dataSize : mNumUnits;
            nvinfer1::Weights gateKernelWeights{dataType, &mWeights->mWeights[0] + kernelOffset, inputSize * mNumUnits};
            nvinfer1::Weights gateBiasWeights{dataType, &mWeights->mWeights[0] + biasOffset, mNumUnits};
            mGateKernelWeights.push_back(std::move(gateKernelWeights));
            mGateBiasWeights.push_back(std::move(gateBiasWeights));
            kernelOffset = kernelOffset + inputSize * mNumUnits * elementSize;
            biasOffset = biasOffset + mNumUnits * elementSize;
        }
    }
    assert(kernelOffset + biasOffset - biasStartOffset == mWeights->mWeights.size());
}

void LSTMDecoder::addToModel(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor* inputEmbeddedData,
    nvinfer1::ITensor** inputStates,
    nvinfer1::ITensor** outputData,
    nvinfer1::ITensor** outputStates)
{
    int beamWidth;
    int inputWidth;
    {
        auto dims = inputEmbeddedData->getDimensions();
        assert(dims.nbDims == 2);
        assert(dims.type[0] == nvinfer1::DimensionType::kINDEX);
        beamWidth = dims.d[0];
        assert(dims.type[1] == nvinfer1::DimensionType::kCHANNEL);
        inputWidth = dims.d[1];
    }

    nvinfer1::ITensor* shuffledInput;
    {
        auto shuffleLayer = network->addShuffle(*inputEmbeddedData);
        assert(shuffleLayer != nullptr);
        shuffleLayer->setName("Reshape input for LSTM decoder");
        nvinfer1::Dims shuffleDims{3, {beamWidth, 1, inputWidth}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kSEQUENCE, nvinfer1::DimensionType::kCHANNEL}};
        shuffleLayer->setReshapeDimensions(shuffleDims);
        shuffledInput = shuffleLayer->getOutput(0);
        assert(shuffledInput != nullptr);
    }

    auto decoderLayer = network->addRNNv2(
        *shuffledInput,
        mNumLayers,
        mNumUnits,
        1,
        nvinfer1::RNNOperation::kLSTM);
    assert(decoderLayer != nullptr);
    decoderLayer->setName("LSTM decoder");

    decoderLayer->setInputMode(nvinfer1::RNNInputMode::kLINEAR);
    decoderLayer->setDirection(nvinfer1::RNNDirection::kUNIDIRECTION);

    std::vector<nvinfer1::RNNGateType> gateOrder({nvinfer1::RNNGateType::kFORGET,
                                                  nvinfer1::RNNGateType::kINPUT,
                                                  nvinfer1::RNNGateType::kCELL,
                                                  nvinfer1::RNNGateType::kOUTPUT});
    for (size_t i = 0; i < mGateKernelWeights.size(); i++)
    {
        // we have 4 + 4 gates
        bool isW = ((i % 8) < 4);
        decoderLayer->setWeightsForGate(i / 8, gateOrder[i % 4], isW, mGateKernelWeights[i]);
        decoderLayer->setBiasForGate(i / 8, gateOrder[i % 4], isW, mGateBiasWeights[i]);
    }

    decoderLayer->setHiddenState(*inputStates[0]);
    decoderLayer->setCellState(*inputStates[1]);
    *outputData = decoderLayer->getOutput(0);
    assert(*outputData != nullptr);

    {
        auto shuffleLayer = network->addShuffle(**outputData);
        assert(shuffleLayer != nullptr);
        shuffleLayer->setName("Reshape output from LSTM decoder");
        nvinfer1::Dims shuffleDims{2, {beamWidth, mNumUnits}, {nvinfer1::DimensionType::kINDEX, nvinfer1::DimensionType::kCHANNEL}};
        shuffleLayer->setReshapeDimensions(shuffleDims);
        auto shuffledOutput = shuffleLayer->getOutput(0);
        assert(shuffledOutput != nullptr);
        *outputData = shuffledOutput;
    }

    // Per layer hidden output
    outputStates[0] = decoderLayer->getOutput(1);
    assert(outputStates[0] != nullptr);

    // Per layer cell output
    outputStates[1] = decoderLayer->getOutput(2);
    assert(outputStates[1] != nullptr);
}

std::vector<nvinfer1::Dims> LSTMDecoder::getStateSizes()
{
    nvinfer1::Dims hiddenStateDims{2, {mNumLayers, mNumUnits}, {nvinfer1::DimensionType::kSPATIAL, nvinfer1::DimensionType::kCHANNEL}};
    nvinfer1::Dims cellStateDims{2, {mNumLayers, mNumUnits}, {nvinfer1::DimensionType::kSPATIAL, nvinfer1::DimensionType::kCHANNEL}};
    return std::vector<nvinfer1::Dims>({hiddenStateDims, cellStateDims});
}

std::string LSTMDecoder::getInfo()
{
    std::stringstream ss;
    ss << "LSTM Decoder, num layers = " << mNumLayers << ", num units = " << mNumUnits;
    return ss.str();
}
} // namespace nmtSample
