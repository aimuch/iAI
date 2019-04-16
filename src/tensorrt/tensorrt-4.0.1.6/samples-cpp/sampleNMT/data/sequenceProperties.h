#ifndef SAMPLE_NMT_SEQUENCE_PROPERTIES_
#define SAMPLE_NMT_SEQUENCE_PROPERTIES_

#include <memory>

namespace nmtSample
{
/** \class SequenceProperties
    *
    * \brief provides encoder/decoder relevant properties of sequences
    *
    */
class SequenceProperties
{
public:
    typedef std::shared_ptr<SequenceProperties> ptr;

    SequenceProperties() = default;

    virtual int getStartSequenceId() = 0;

    virtual int getEndSequenceId() = 0;

    virtual ~SequenceProperties() = default;
};
}

#endif // SAMPLE_NMT_SEQUENCE_PROPERTIES_
