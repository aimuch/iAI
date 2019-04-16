#ifndef SAMPLE_NMT_CONTEXT_
#define SAMPLE_NMT_CONTEXT_

#include <memory>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Context
    *
    * \brief calculates context vector from raw alignment scores and memory states
    *
    */
class Context : public Component
{
public:
    typedef std::shared_ptr<Context> ptr;

    Context() = default;

    /**
        * \brief add the context vector calculation to the network
        */
    void addToModel(
        nvinfer1::INetworkDefinition* network,
        nvinfer1::ITensor* actualInputSequenceLengths,
        nvinfer1::ITensor* memoryStates,
        nvinfer1::ITensor* alignmentScores,
        nvinfer1::ITensor** contextOutput);

    std::string getInfo() override;

    ~Context() override = default;
};
}

#endif // SAMPLE_NMT_CONTEXT_
