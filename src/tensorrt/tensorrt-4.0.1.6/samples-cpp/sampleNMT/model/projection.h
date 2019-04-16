#ifndef SAMPLE_NMT_PROJECTION_
#define SAMPLE_NMT_PROJECTION_

#include <memory>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Projection
    *
    * \brief calculates raw logits
    *
    */
class Projection : public Component
{
public:
    typedef std::shared_ptr<Projection> ptr;

    Projection() = default;

    /**
        * \brief add raw logits to the network
        */
    virtual void addToModel(
        nvinfer1::INetworkDefinition* network,
        nvinfer1::ITensor* input,
        nvinfer1::ITensor** outputLogits)
        = 0;

    /**
        * \brief get the size of raw logits vector
        */
    virtual int getOutputSize() = 0;

    ~Projection() override = default;
};
}

#endif // SAMPLE_NMT_PROJECTION_
