#include "textReader.h"

#include <algorithm>
#include <clocale>
#include <fstream>
#include <sstream>

namespace nmtSample
{
TextReader::TextReader(std::shared_ptr<std::istream> textInput, Vocabulary::ptr vocabulary)
    : mInput(textInput)
    , mVocabulary(vocabulary)
{
}

int TextReader::read(
    int samplesToRead,
    int maxInputSequenceLength,
    int* hInputData,
    int* hActualInputSequenceLengths)
{
    std::setlocale(LC_ALL, "en_US.UTF-8");
    std::string line;

    int lineCounter = 0;
    while (lineCounter < samplesToRead && std::getline(*mInput, line))
    {
        std::istringstream ss(line);
        std::string token;
        int tokenCounter = 0;
        while ((ss >> token) && (tokenCounter < maxInputSequenceLength))
        {
            hInputData[maxInputSequenceLength * lineCounter + tokenCounter] = mVocabulary->getId(token);
            tokenCounter++;
        }

        hActualInputSequenceLengths[lineCounter] = tokenCounter;

        // Fill unused values with valid vocabulary ID, it doesn't necessary have to be eos
        std::fill(hInputData + maxInputSequenceLength * lineCounter + tokenCounter, hInputData + maxInputSequenceLength * (lineCounter + 1), mVocabulary->getEndSequenceId());

        lineCounter++;
    }
    return lineCounter;
}

void TextReader::reset()
{
    mInput->seekg(0, mInput->beg);
}

std::string TextReader::getInfo()
{
    std::stringstream ss;
    ss << "Text Reader, vocabulary size = " << mVocabulary->getSize();
    return ss.str();
}
}
