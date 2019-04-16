#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include <vector>
#include <assert.h>
#include <algorithm>
#include "NvInfer.h"

std::string locateFile(const std::string& input);

class BatchStream
{
public:
	BatchStream(int batchSize, int maxBatches) : mBatchSize(batchSize), mMaxBatches(maxBatches)
	{
		FILE* file = fopen(locateFile(std::string("batches/batch0")).c_str(), "rb");
		int d[4];
		fread(d, sizeof(int), 4, file);
		mDims = nvinfer1::DimsNCHW{ d[0], d[1], d[2], d[3] };
		fclose(file);
		mImageSize = mDims.c()*mDims.h()*mDims.w();
		mBatch.resize(mBatchSize*mImageSize, 0);
		mLabels.resize(mBatchSize, 0);
		mFileBatch.resize(mDims.n()*mImageSize, 0);
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
			std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
		}
		mBatchCount++;
		return true;
	}

	void skip(int skipCount)
	{
		if (mBatchSize >= mDims.n() && mBatchSize%mDims.n() == 0 && mFileBatchPos == mDims.n())
		{
			mFileCount += skipCount * mBatchSize / mDims.n();
			return;
		}

		int x = mBatchCount;
		for (int i = 0; i < skipCount; i++)
			next();
		mBatchCount = x;
	}

	float *getBatch() { return &mBatch[0]; }
	float *getLabels() { return &mLabels[0]; }
	int getBatchesRead() const { return mBatchCount; }
	int getBatchSize() const { return mBatchSize; }
	nvinfer1::DimsNCHW getDims() const { return mDims; }
private:
	float* getFileBatch() { return &mFileBatch[0]; }
	float* getFileLabels() { return &mFileLabels[0]; }

	bool update()
	{
		std::string inputFileName = locateFile(std::string("batches/batch") + std::to_string(mFileCount++));
		FILE * file = fopen(inputFileName.c_str(), "rb");
		if (!file)
			return false;

		int d[4];
		fread(d, sizeof(int), 4, file);
		assert(mDims.n() == d[0] && mDims.c() == d[1] && mDims.h() == d[2] && mDims.w() == d[3]);

		size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.n()*mImageSize, file);
		size_t readLabelCount = fread(getFileLabels(), sizeof(float), mDims.n(), file);;
		assert(readInputCount == size_t(mDims.n()*mImageSize) && readLabelCount == size_t(mDims.n()));

		fclose(file);
		mFileBatchPos = 0;
		return true;
	}

	int mBatchSize{ 0 };
	int mMaxBatches{ 0 };
	int mBatchCount{ 0 };

	int mFileCount{ 0 }, mFileBatchPos{ 0 };
	int mImageSize{ 0 };

	nvinfer1::DimsNCHW mDims;
	std::vector<float> mBatch;
	std::vector<float> mLabels;
	std::vector<float> mFileBatch;
	std::vector<float> mFileLabels;
};


#endif
