#ifndef SAMPLE_NMT_TEXT_WRITER_
#define SAMPLE_NMT_TEXT_WRITER_

#include <memory>
#include <ostream>

#include "dataWriter.h"
#include "vocabulary.h"

namespace nmtSample
{
/** \class TextReader
    *
    * \brief writes sequences of data into output stream
    *
    */
class TextWriter : public DataWriter
{
public:
    TextWriter(std::shared_ptr<std::ostream> textOnput, Vocabulary::ptr vocabulary);

    void write(
        const int* hOutputData,
        int actualOutputSequenceLength,
        int actualInputSequenceLength) override;

    void initialize() override;

    void finalize() override;

    std::string getInfo() override;

    ~TextWriter() override = default;

private:
    std::shared_ptr<std::ostream> mOutput;
    Vocabulary::ptr mVocabulary;
};
}

#endif // SAMPLE_NMT_TEXT_WRITER_
