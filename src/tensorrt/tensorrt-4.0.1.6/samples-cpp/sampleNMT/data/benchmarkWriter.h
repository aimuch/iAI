#ifndef SAMPLE_NMT_BENCHMARK_WRITER_
#define SAMPLE_NMT_BENCHMARK_WRITER_

#include <chrono>
#include <memory>

#include "dataWriter.h"

namespace nmtSample
{
/** \class BenchmarkWriter
    *
    * \brief all it does is to measure the performance of sequence generation
    *
    */
class BenchmarkWriter : public DataWriter
{
public:
    BenchmarkWriter();

    void write(
        const int* hOutputData,
        int actualOutputSequenceLength,
        int actualInputSequenceLength) override;

    void initialize() override;

    void finalize() override;

    std::string getInfo() override;

    ~BenchmarkWriter() override = default;

private:
    int mSampleCount;
    int mInputTokenCount;
    int mOutputTokenCount;
    std::chrono::high_resolution_clock::time_point mStartTS;
};
}

#endif // SAMPLE_NMT_BENCHMARK_WRITER_
