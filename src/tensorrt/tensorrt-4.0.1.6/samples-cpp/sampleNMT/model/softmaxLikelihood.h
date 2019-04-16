#ifndef SAMPLE_NMT_SOFTMAX_LIKELIHOOD_
#define SAMPLE_NMT_SOFTMAX_LIKELIHOOD_

#include "NvInfer.h"
#include "likelihood.h"

namespace nmtSample
{
/** \class SoftmaxLikelihood
    *
    * \brief calculates softmax likelihood and TopK indices for the raw input logits
    *
    */
class SoftmaxLikelihood : public Likelihood
{
private:
    class SoftmaxLikelihoodCombinationOperator : public LikelihoodCombinationOperator
    {
    public:
        SoftmaxLikelihoodCombinationOperator() = default;

        float combine(float rayLikelihood, float optionLikelihood) const override;

        float init() const override;

        float smallerThanMinimalLikelihood() const override;

        ~SoftmaxLikelihoodCombinationOperator() override = default;
    };

public:
    SoftmaxLikelihood() = default;

    LikelihoodCombinationOperator::ptr getLikelihoodCombinationOperator() const override;

    void addToModel(
        nvinfer1::INetworkDefinition* network,
        int beamWidth,
        nvinfer1::ITensor* inputLogits,
        nvinfer1::ITensor* inputLikelihoods,
        nvinfer1::ITensor** newCombinedLikelihoods,
        nvinfer1::ITensor** newRayOptionIndices,
        nvinfer1::ITensor** newVocabularyIndices) override;

    std::string getInfo() override;

    ~SoftmaxLikelihood() override = default;
};
}

#endif // SAMPLE_NMT_SOFTMAX_LIKELIHOOD_
