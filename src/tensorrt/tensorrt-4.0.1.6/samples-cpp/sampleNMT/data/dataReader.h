#ifndef SAMPLE_NMT_DATA_READER_
#define SAMPLE_NMT_DATA_READER_

#include <memory>

#include "../component.h"

namespace nmtSample
{
/** \class DataReader
    *
    * \brief reader of sequences of data
    *
    */
class DataReader : public Component
{
public:
    typedef std::shared_ptr<DataReader> ptr;

    DataReader() = default;

    /**
        * \brief reads the batch of smaples/sequences
        *
        * \return the actual number of samples read
        */
    virtual int read(
        int samplesToRead,
        int maxInputSequenceLength,
        int* hInputData,
        int* hActualInputSequenceLengths)
        = 0;

    /**
        * \brief Reset the reader position, the data reader is ready to read the data from th ebeginning again after the function returns
        */
    virtual void reset() = 0;

    ~DataReader() override = default;
};
}

#endif // SAMPLE_NMT_DATA_READER_
