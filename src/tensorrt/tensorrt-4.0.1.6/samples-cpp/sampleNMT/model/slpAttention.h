#ifndef SAMPLE_NMT_SLP_ATTENTION_
#define SAMPLE_NMT_SLP_ATTENTION_

#include "attention.h"

#include "componentWeights.h"

namespace nmtSample
{
/** \class SLPAttention
    *
    * \brief Linear attention calculation
    *
    * Calculates attention vector by concatinating input from the decoder with context vector
    * and projecting the result into attention space by multiplying with weight matrix  
    *
    */
class SLPAttention : public Attention
{
public:
    SLPAttention(ComponentWeights::ptr weights);

    void addToModel(
        nvinfer1::INetworkDefinition* network,
        nvinfer1::ITensor* inputFromDecoder,
        nvinfer1::ITensor* context,
        nvinfer1::ITensor** attentionOutput) override;

    int getAttentionSize() override;

    std::string getInfo() override;

protected:
    ComponentWeights::ptr mWeights;
    nvinfer1::Weights mKernelWeights;
    int mInputChannelCount;
    int mOutputChannelCount;
};
}

#endif // SAMPLE_NMT_SLP_ATTENTION_
