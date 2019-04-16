#include "benchmarkWriter.h"

#include <iostream>

namespace nmtSample
{
BenchmarkWriter::BenchmarkWriter()
    : mSampleCount(0)
    , mInputTokenCount(0)
    , mOutputTokenCount(0)
    , mStartTS(std::chrono::high_resolution_clock::now())
{
}

void BenchmarkWriter::write(
    const int* hOutputData,
    int actualOutputSequenceLength,
    int actualInputSequenceLength)
{
    ++mSampleCount;
    mInputTokenCount += actualInputSequenceLength;
    mOutputTokenCount += actualOutputSequenceLength;
}

void BenchmarkWriter::initialize()
{
    mStartTS = std::chrono::high_resolution_clock::now();
}

void BenchmarkWriter::finalize()
{
    std::chrono::duration<float> sec = std::chrono::high_resolution_clock::now() - mStartTS;
    int totalTokenCount = mInputTokenCount + mOutputTokenCount;
    std::cout << mSampleCount << " sequences generated in " << sec.count() << " seconds, " << (mSampleCount / sec.count()) << " samples/sec" << std::endl;
    std::cout << totalTokenCount << " tokens processed (source and destination), " << (totalTokenCount / sec.count()) << " tokens/sec" << std::endl;
}

std::string BenchmarkWriter::getInfo()
{
    return "Benchmark Writer";
}
}
