#ifndef SAMPLE_NMT_LIKELIHOOD_
#define SAMPLE_NMT_LIKELIHOOD_

#include <memory>

#include "../component.h"
#include "NvInfer.h"
#include "likelihoodCombinationOperator.h"

namespace nmtSample
{
/** \class Likelihood
    *
    * \brief calculates likelihood and TopK indices for the raw input logits
    *
    */
class Likelihood : public Component
{
public:
    typedef std::shared_ptr<Likelihood> ptr;

    Likelihood() = default;

    virtual LikelihoodCombinationOperator::ptr getLikelihoodCombinationOperator() const = 0;

    /**
        * \brief add calculation of likelihood and TopK indices to the network
        */
    virtual void addToModel(
        nvinfer1::INetworkDefinition* network,
        int beamWidth,
        nvinfer1::ITensor* inputLogits,
        nvinfer1::ITensor* inputLikelihoods,
        nvinfer1::ITensor** newCombinedLikelihoods,
        nvinfer1::ITensor** newRayOptionIndices,
        nvinfer1::ITensor** newVocabularyIndices)
        = 0;

    ~Likelihood() override = default;
};
}

#endif // SAMPLE_NMT_LIKELIHOOD_
