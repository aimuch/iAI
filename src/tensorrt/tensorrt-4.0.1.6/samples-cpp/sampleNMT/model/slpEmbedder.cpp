#include "slpEmbedder.h"

#include <cassert>
#include <sstream>

namespace nmtSample
{
SLPEmbedder::SLPEmbedder(ComponentWeights::ptr weights)
    : mWeights(weights)
{
    // please refer to chpt_to_bin.py for the details on the format
    assert(mWeights->mMetaData.size() >= 3);
    mKernelWeights.type = static_cast<nvinfer1::DataType>(mWeights->mMetaData[0]);
    assert(mKernelWeights.type == nvinfer1::DataType::kFLOAT);
    mNumInputs = mWeights->mMetaData[1];
    mNumOutputs = mWeights->mMetaData[2];

    mKernelWeights.values = (void*) (&mWeights->mWeights[0]);
    mKernelWeights.count = mNumInputs * mNumOutputs;
}

void SLPEmbedder::addToModel(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor* input,
    nvinfer1::ITensor** output)
{
    nvinfer1::Dims weightDims{2, {mNumInputs, mNumOutputs}, {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kCHANNEL}};
    auto constLayer = network->addConstant(weightDims, mKernelWeights);
    assert(constLayer != nullptr);
    constLayer->setName("Embedding matrix");
    auto weights = constLayer->getOutput(0);
    assert(weights != nullptr);

    auto gatherLayer = network->addGather(*weights, *input, 0);
    assert(gatherLayer != nullptr);
    gatherLayer->setName("Gather in embedding");
    *output = gatherLayer->getOutput(0);
    assert(*output != nullptr);
}

int SLPEmbedder::getInputDimensionSize()
{
    return mNumInputs;
}

std::string SLPEmbedder::getInfo()
{
    std::stringstream ss;
    ss << "SLP Embedder, num inputs = " << mNumInputs << ", num outputs = " << mNumOutputs;
    return ss.str();
}
}
