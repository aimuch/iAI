#ifndef BATCH_STREAM_PPM_H
#define BATCH_STREAM_PPM_H
#include <vector>
#include <assert.h>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include "NvInfer.h"
#include "common.h"

std::string locateFile(const std::string& input);

static constexpr int INPUT_C = 3;
static constexpr int INPUT_H = 300;
static constexpr int INPUT_W = 300;

const char* INPUT_BLOB_NAME = "Input";

// Simple PPM (portable pixel map) reader.
template <int C, int H, int W>
void readPPMFile(const std::string& filename, samples_common::PPM<C, H, W>& ppm)
{
    ppm.fileName = filename;
    std::ifstream infile(locateFile(filename), std::ifstream::binary);
    infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

class BatchStream
{
public:
	BatchStream(int batchSize, int maxBatches) : mBatchSize(batchSize), mMaxBatches(maxBatches)
	{
		mDims = nvinfer1::DimsNCHW{batchSize, 3, 300, 300 };
		mImageSize = mDims.c() * mDims.h() * mDims.w();
		mBatch.resize(mBatchSize * mImageSize, 0);
		mLabels.resize(mBatchSize, 0);
		mFileBatch.resize(mDims.n() * mImageSize, 0);
		mFileLabels.resize(mDims.n(), 0);
		reset(0);
	}

	void reset(int firstBatch)
	{
		mBatchCount = 0;
		mFileCount = 0;
		mFileBatchPos = mDims.n();
		skip(firstBatch);
	}

	bool next()
	{
		if (mBatchCount == mMaxBatches)
			return false;

		for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
		{
			assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.n());
			if (mFileBatchPos == mDims.n() && !update())
				return false;

			// copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
			csize = std::min(mBatchSize - batchPos, mDims.n() - mFileBatchPos);
			std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
		}
		mBatchCount++;
		return true;
	}

	void skip(int skipCount)
	{
		if (mBatchSize >= mDims.n() && mBatchSize % mDims.n() == 0 && mFileBatchPos == mDims.n())
		{
			mFileCount += skipCount * mBatchSize / mDims.n();
			return;
		}

		int x = mBatchCount;
		for (int i = 0; i < skipCount; i++)
			next();
		mBatchCount = x;
	}

	float *getBatch() { return mBatch.data(); }
	float *getLabels() { return mLabels.data(); }
	int getBatchesRead() const { return mBatchCount; }
	int getBatchSize() const { return mBatchSize; }
	nvinfer1::DimsNCHW getDims() const { return mDims; }
private:
	float* getFileBatch() { return mFileBatch.data(); }
	float* getFileLabels() { return mFileLabels.data(); }

	bool update()
	{
        std::vector<std::string> fNames;

	    std::ifstream file(locateFile("list.txt"));
        if(file)
        {
            std::cout  << "Batch #" << mFileCount << "\n";
            file.seekg(((mBatchCount * mBatchSize))*7);
        }
        for(int i = 1; i <= mBatchSize; i++)
        {
            std::string sName;
            std::getline(file, sName);
            sName = sName + ".ppm";

            std::cout << "Calibrating with file " << sName << std::endl;
            fNames.emplace_back(sName);
        }
        mFileCount++;

        std::vector<samples_common::PPM<INPUT_C, INPUT_H, INPUT_W>> ppms(fNames.size());
        for (uint i = 0; i < fNames.size(); ++i)
        {
            readPPMFile(fNames[i], ppms[i]);
        }
        std::vector<float> data(samples_common::volume(mDims));

        long int volChl = mDims.h() * mDims.w();

        for (int i = 0, volImg = mDims.c() * mDims.h() * mDims.w(); i < mBatchSize; ++i)
        {
            for (int c = 0; c < mDims.c(); ++c)
            {
                for (int j = 0; j < volChl; ++j)
                {
                    data[i * volImg + c * volChl + j] = (2.0 / 255.0) * float(ppms[i].buffer[j * mDims.c() + c]) - 1.0;
                }
            }
        }

        std::copy_n(data.data(), mDims.n() * mImageSize, getFileBatch());

		mFileBatchPos = 0;
		return true;
	}

	int mBatchSize{0};
	int mMaxBatches{0};
	int mBatchCount{0};

	int mFileCount{0}, mFileBatchPos{0};
	int mImageSize{0};

	nvinfer1::DimsNCHW mDims;
	std::vector<float> mBatch;
	std::vector<float> mLabels;
	std::vector<float> mFileBatch;
	std::vector<float> mFileLabels;
};

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator
{
public:
    Int8EntropyCalibrator(BatchStream& stream, int firstBatch, std::string calibrationTableName, bool readCache = true)
        : mStream(stream),
        mCalibrationTableName(std::move(calibrationTableName)),
        mReadCache(readCache)
    {
    	nvinfer1::DimsNCHW dims = mStream.getDims();
        mInputCount = samples_common::volume(dims);
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }

    virtual ~Int8EntropyCalibrator()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override { return mStream.getBatchSize(); }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
            return false;

        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], INPUT_BLOB_NAME));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    BatchStream mStream;
    std::string mCalibrationTableName;
    bool mReadCache{true};

    size_t mInputCount;
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};
#endif
