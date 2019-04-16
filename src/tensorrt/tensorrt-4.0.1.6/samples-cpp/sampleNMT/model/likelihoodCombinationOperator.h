#ifndef SAMPLE_NMT_LIKELIHOOD_COMBINATION_
#define SAMPLE_NMT_LIKELIHOOD_COMBINATION_

#include <memory>

namespace nmtSample
{
class LikelihoodCombinationOperator
{
public:
    typedef std::shared_ptr<LikelihoodCombinationOperator> ptr;

    // The  return value should be less or equal to rayLikelihood
    virtual float combine(float rayLikelihood, float optionLikelihood) const = 0;

    virtual float init() const = 0;

    virtual float smallerThanMinimalLikelihood() const = 0;

    virtual ~LikelihoodCombinationOperator() = default;

protected:
    LikelihoodCombinationOperator() = default;
};
}

#endif // SAMPLE_NMT_LIKELIHOOD_COMBINATION_
