#ifndef SAMPLE_NMT_SLP_EMBEDDER_
#define SAMPLE_NMT_SLP_EMBEDDER_

#include "embedder.h"

#include "componentWeights.h"

#include "NvInfer.h"

#include <vector>

namespace nmtSample
{
/** \class SLPEmbedder
    *
    * \brief selects the embedding vector from the weight matrix using index provided in the input
    *
    */
class SLPEmbedder : public Embedder
{
public:
    SLPEmbedder(ComponentWeights::ptr weights);

    void addToModel(
        nvinfer1::INetworkDefinition* network,
        nvinfer1::ITensor* input,
        nvinfer1::ITensor** output) override;

    int getInputDimensionSize() override;

    std::string getInfo() override;

    ~SLPEmbedder() override = default;

protected:
    ComponentWeights::ptr mWeights;
    nvinfer1::Weights mKernelWeights;
    int mNumInputs;
    int mNumOutputs;
};
}

#endif // SAMPLE_NMT_SLP_EMBEDDER_
