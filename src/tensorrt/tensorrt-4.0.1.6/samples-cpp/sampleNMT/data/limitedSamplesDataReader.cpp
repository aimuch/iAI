#include "limitedSamplesDataReader.h"

#include <sstream>

namespace nmtSample
{
LimitedSamplesDataReader::LimitedSamplesDataReader(int maxSamplesToRead, DataReader::ptr originalDataReader)
    : gMaxSamplesToRead(maxSamplesToRead)
    , gOriginalDataReader(originalDataReader)
    , gCurrentPosition(0)
{
}

int LimitedSamplesDataReader::read(
    int samplesToRead,
    int maxInputSequenceLength,
    int* hInputData,
    int* hActualInputSequenceLengths)
{
    int limitedSmplesToRead = std::min(samplesToRead, std::max(gMaxSamplesToRead - gCurrentPosition, 0));
    int samplesRead = gOriginalDataReader->read(limitedSmplesToRead, maxInputSequenceLength, hInputData, hActualInputSequenceLengths);
    gCurrentPosition += samplesRead;
    return samplesRead;
}

void LimitedSamplesDataReader::reset()
{
    gOriginalDataReader->reset();
    gCurrentPosition = 0;
}

std::string LimitedSamplesDataReader::getInfo()
{
    std::stringstream ss;
    ss << "Limited Samples Reader, max samples = " << gMaxSamplesToRead << ", original reader info: " << gOriginalDataReader->getInfo();
    return ss.str();
}
}