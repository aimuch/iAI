#ifndef SAMPLE_NMT_DATA_WRITER_
#define SAMPLE_NMT_DATA_WRITER_

#include <memory>
#include <string>

#include "../component.h"
#include "vocabulary.h"

namespace nmtSample
{
/** \class DataWriter
    *
    * \brief writer of sequences of data
    *
    */
class DataWriter : public Component
{
public:
    typedef std::shared_ptr<DataWriter> ptr;

    DataWriter() = default;

    /**
        * \brief write the generated sequence
        */
    virtual void write(
        const int* hOutputData,
        int actualOutputSequenceLength,
        int actualInputSequenceLength)
        = 0;

    /**
        * \brief it is called right before inference starts
        */
    virtual void initialize() = 0;

    /**
        * \brief it is called right after inference ends
        */
    virtual void finalize() = 0;

    ~DataWriter() override = default;

protected:
    static std::string generateText(int sequenceLength, const int* currentOutputData, Vocabulary::ptr vocabulary);
};
}

#endif // SAMPLE_NMT_DATA_WRITER_
