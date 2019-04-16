#include "slpAttention.h"

#include <cassert>
#include <sstream>

namespace nmtSample
{
SLPAttention::SLPAttention(ComponentWeights::ptr weights)
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

void SLPAttention::addToModel(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor* inputFromDecoder,
    nvinfer1::ITensor* context,
    nvinfer1::ITensor** attentionOutput)
{
    nvinfer1::ITensor* inputTensors[] = {inputFromDecoder, context};
    auto concatLayer = network->addConcatenation(inputTensors, 2);
    assert(concatLayer != nullptr);
    concatLayer->setName("Concatinate decoder output and context");
    concatLayer->setAxis(1);
    auto concatinatedTensor = concatLayer->getOutput(0);
    assert(concatinatedTensor != nullptr);

    nvinfer1::Dims weightDims{2, {mInputChannelCount, mOutputChannelCount}, {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kCHANNEL}};
    auto constLayer = network->addConstant(weightDims, mKernelWeights);
    assert(constLayer != nullptr);
    constLayer->setName("Attention Matrix");
    auto weights = constLayer->getOutput(0);
    assert(weights != nullptr);

    auto mmLayer = network->addMatrixMultiply(
        *concatinatedTensor,
        false,
        *weights,
        false);
    assert(mmLayer != nullptr);
    mmLayer->setName("Attention Matrix Multiply");
    *attentionOutput = mmLayer->getOutput(0);
    assert(*attentionOutput != nullptr);
}

int SLPAttention::getAttentionSize()
{
    return mOutputChannelCount;
}

std::string SLPAttention::getInfo()
{
    std::stringstream ss;
    ss << "SLP Attention, num inputs = " << mInputChannelCount << ", num outputs = " << mOutputChannelCount;
    return ss.str();
}
}
