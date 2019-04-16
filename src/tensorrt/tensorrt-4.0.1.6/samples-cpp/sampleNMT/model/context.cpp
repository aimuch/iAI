#include "context.h"

#include <cassert>
#include <sstream>

namespace nmtSample
{
void Context::addToModel(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor* actualInputSequenceLengths,
    nvinfer1::ITensor* memoryStates,
    nvinfer1::ITensor* alignmentScores,
    nvinfer1::ITensor** contextOutput)
{
    auto raggedSoftmaxLayer = network->addRaggedSoftMax(*alignmentScores, *actualInputSequenceLengths);
    assert(raggedSoftmaxLayer != nullptr);
    raggedSoftmaxLayer->setName("Context Ragged Softmax");
    auto softmaxTensor = raggedSoftmaxLayer->getOutput(0);
    assert(softmaxTensor != nullptr);

    auto mmLayer = network->addMatrixMultiply(
        *softmaxTensor,
        false,
        *memoryStates,
        false);
    assert(mmLayer != nullptr);
    mmLayer->setName("Context Matrix Multiply");
    *contextOutput = mmLayer->getOutput(0);
    assert(*contextOutput != nullptr);
}

std::string Context::getInfo()
{
    return "Ragged softmax + Batch GEMM";
}
}
