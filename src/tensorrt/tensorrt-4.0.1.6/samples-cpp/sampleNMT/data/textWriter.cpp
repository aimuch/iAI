#include "textWriter.h"

#include <iostream>
#include <regex>
#include <sstream>

namespace nmtSample
{
TextWriter::TextWriter(std::shared_ptr<std::ostream> textOnput, Vocabulary::ptr vocabulary)
    : mOutput(textOnput)
    , mVocabulary(vocabulary)
{
}

void TextWriter::write(
    const int* hOutputData,
    int actualOutputSequenceLength,
    int actualInputSequenceLength)
{
    // if clean and handle BPE outputs is required
    *mOutput << DataWriter::generateText(actualOutputSequenceLength, hOutputData, mVocabulary) << "\n";
}

void TextWriter::initialize()
{
}

void TextWriter::finalize()
{
}

std::string TextWriter::getInfo()
{
    std::stringstream ss;
    ss << "Text Writer, vocabulary size = " << mVocabulary->getSize();
    return ss.str();
}
}
