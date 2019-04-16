#include <sstream>

#include "dataWriter.h"

namespace nmtSample
{
std::string DataWriter::generateText(int sequenceLength, const int* currentOutputData, Vocabulary::ptr vocabulary)
{
    // if clean and handle BPE outputs is required
    std::string delimiter = "@@";
    size_t delimiterSize = delimiter.size();
    std::stringstream sentence;
    std::string word("");
    const char* wordDelimiter = "";
    for (int i = 0; i < sequenceLength; ++i)
    {
        int id = currentOutputData[i];
        if (id != vocabulary->getEndSequenceId())
        {
            std::string token = vocabulary->getToken(id);
            if ((token.size() >= delimiterSize) && (token.compare(token.size() - delimiterSize, delimiterSize, delimiter) == 0))
            {
                word = word + token.erase(token.size() - delimiterSize, delimiterSize);
            }
            else
            {
                word = word + token;
                sentence << wordDelimiter;
                sentence << word;
                word = "";
                wordDelimiter = " ";
            }
        }
    }
    return sentence.str();
}
}