#ifndef SAMPLE_NMT_EMBEDDER_
#define SAMPLE_NMT_EMBEDDER_

#include <memory>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Embedder
    *
    * \brief projects 1-hot vectors (represented as a vector with indices) into dense embedding space
    *
    */
class Embedder : public Component
{
public:
    typedef std::shared_ptr<Embedder> ptr;

    Embedder() = default;

    /**
        * \brief add the embedding vector calculation to the network
        */
    virtual void addToModel(
        nvinfer1::INetworkDefinition* network,
        nvinfer1::ITensor* input,
        nvinfer1::ITensor** output)
        = 0;

    /**
        * \brief get the upper bound for the possible values of indices 
        */
    virtual int getInputDimensionSize() = 0;

    ~Embedder() override = default;
};
}

#endif // SAMPLE_NMT_EMBEDDER_
