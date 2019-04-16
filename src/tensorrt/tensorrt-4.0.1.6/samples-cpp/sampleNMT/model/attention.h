#ifndef SAMPLE_NMT_ATTENTION_
#define SAMPLE_NMT_ATTENTION_

#include <memory>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Attention
    *
    * \brief calculates attention vector from context and decoder output vectors 
    *
    */
class Attention : public Component
{
public:
    typedef std::shared_ptr<Attention> ptr;

    Attention() = default;

    /**
        * \brief add the attention vector calculation to the network
        */
    virtual void addToModel(
        nvinfer1::INetworkDefinition* network,
        nvinfer1::ITensor* inputFromDecoder,
        nvinfer1::ITensor* context,
        nvinfer1::ITensor** attentionOutput)
        = 0;

    /**
        * \brief get the size of the attention vector
        */
    virtual int getAttentionSize() = 0;

    ~Attention() override = default;
};
}

#endif // SAMPLE_NMT_ATTENTION_
