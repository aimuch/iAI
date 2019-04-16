#ifndef LEGACY_CALIBRATOR_H
#define LEGACY_CALIBRATOR_H

#include <iostream>
#include "NvInfer.h"
#include "BatchStream.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <iterator>

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

class Int8LegacyCalibrator : public nvinfer1::IInt8LegacyCalibrator
{
public:
	Int8LegacyCalibrator(BatchStream& stream, int firstBatch, double cutoff, double quantile, bool readCache = true)
		: mStream(stream), mFirstBatch(firstBatch), mReadCache(readCache)
	{
		using namespace nvinfer1;
		DimsNCHW dims = mStream.getDims();
		mInputCount = mStream.getBatchSize() * dims.c() * dims.h() * dims.w();
		CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
		reset(cutoff, quantile);
	}

	virtual ~Int8LegacyCalibrator()
	{
		CHECK(cudaFree(mDeviceInput));
	}

	int getBatchSize() const override { return mStream.getBatchSize(); }
	double getQuantile() const override { return mQuantile; }
	double getRegressionCutoff() const override { return mCutoff; }

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
		if (!mStream.next())
			return false;

		CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
		bindings[0] = mDeviceInput;
		return true;
	}

	const void* readCalibrationCache(size_t& length) override
	{
		mCalibrationCache.clear();
		std::ifstream input(locateFile("CalibrationTable"), std::ios::binary);
		input >> std::noskipws;
		if (mReadCache && input.good())
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

		length = mCalibrationCache.size();
		return length ? &mCalibrationCache[0] : nullptr;
	}

	void writeCalibrationCache(const void* cache, size_t length) override
	{
		std::ofstream output(locateFile("CalibrationTable"), std::ios::binary);
		output.write(reinterpret_cast<const char*>(cache), length);
	}

	const void* readHistogramCache(size_t& length) override
	{
		length = mHistogramCache.size();
		return length ? &mHistogramCache[0] : nullptr;
	}

	void writeHistogramCache(const void* cache, size_t length) override
	{
		mHistogramCache.clear();
		std::copy_n(reinterpret_cast<const char*>(cache), length, std::back_inserter(mHistogramCache));
	}

	void reset(double cutoff, double quantile)
	{
		mCutoff = cutoff;
		mQuantile = quantile;
		mStream.reset(mFirstBatch);
	}

private:
	BatchStream mStream;
	int mFirstBatch;
	double mCutoff, mQuantile;
	bool mReadCache{ true };

	size_t mInputCount;
	void* mDeviceInput{ nullptr };
	std::vector<char> mCalibrationCache, mHistogramCache;
};

struct CalibrationParameters
{
	const char* networkName;
	double cutoff;
	double quantileIndex;
};

CalibrationParameters gCalibrationTable[] =
{
	{ "alexnet", 0.6, 7.0 },
	{ "vgg19", 0.5, 5 },
	{ "googlenet", 1, 8.0 },
	{ "resnet-50", 0.61, 2.0 },
	{ "resnet-101", 0.51, 2.5 },
	{ "resnet-152", 0.4, 5.0 }
};

static const int gCalibrationTableSize = sizeof(gCalibrationTable) / sizeof(CalibrationParameters);

double quantileFromIndex(double quantileIndex)
{
	return 1 - pow(10, -quantileIndex);
}

static const int CAL_BATCH_SIZE = 50;
static const int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;					// calibrate over images 0-500
static const int FIRST_CAL_SCORE_BATCH = 100, NB_CAL_SCORE_BATCHES = 100;	// score over images 5000-10000


void searchCalibrations(double firstCutoff, double cutoffIncrement, int nbCutoffs,
	double firstQuantileIndex, double quantileIndexIncrement, int nbQuantiles,
	float& bestScore, double& bestCutoff, double& bestQuantileIndex, Int8LegacyCalibrator& calibrator)
{
	std::pair<float, float> scoreModel(int batchSize, int firstBatch, int nbScoreBatches, nvinfer1::DataType type, nvinfer1::IInt8Calibrator* calibrator, bool quiet);

	for (int i = 0; i < nbCutoffs; i++)
	{
		for (int j = 0; j < nbQuantiles; j++)
		{
			double cutoff = firstCutoff + double(i) * cutoffIncrement, quantileIndex = firstQuantileIndex + double(j) * quantileIndexIncrement;
			calibrator.reset(cutoff, quantileFromIndex(quantileIndex));
			float score = scoreModel(CAL_BATCH_SIZE, FIRST_CAL_SCORE_BATCH, NB_CAL_SCORE_BATCHES, nvinfer1::DataType::kINT8, &calibrator, true).first;		// score the model in quiet mode

			std::cout << "Score: " << score << " (cutoff = " << cutoff << ", quantileIndex = " << quantileIndex << ")" << std::endl;
			if (score > bestScore)
				bestScore = score, bestCutoff = cutoff, bestQuantileIndex = quantileIndex;
		}
	}
}

void searchCalibrations(double& bestCutoff, double&bestQuantileIndex)
{
	float bestScore = std::numeric_limits<float>::lowest();
	bestCutoff = 0;
	bestQuantileIndex = 0;

	std::cout << "searching calibrations" << std::endl;
	BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);
	Int8LegacyCalibrator calibrator(calibrationStream, 0, quantileFromIndex(0), false);					// force calibration by ignoring region cache

	searchCalibrations(1, 0, 1, 2, 1, 7, bestScore, bestCutoff, bestQuantileIndex, calibrator);			// search the space with cutoff = 1 (i.e. max'ing over the histogram)
	searchCalibrations(0.4, 0.05, 7, 2, 1, 7, bestScore, bestCutoff, bestQuantileIndex, calibrator);	// search the space with cutoff = 0.4 to 0.7 (inclusive)

	// narrow in: if our best score is at cutoff 1 then search over quantiles, else over both dimensions
	if (bestScore == 1)
		searchCalibrations(1, 0, 1, bestQuantileIndex - 0.5, 0.1, 11, bestScore, bestCutoff, bestQuantileIndex, calibrator);
	else
		searchCalibrations(bestCutoff - 0.04, 0.01, 9, bestQuantileIndex - 0.5, 0.1, 11, bestScore, bestCutoff, bestQuantileIndex, calibrator);
	std::cout << "\n\nBest score: " << bestScore << " (cutoff = " << bestCutoff << ", quantileIndex = " << bestQuantileIndex << ")" << std::endl;
}

std::pair<double, double> getQuantileAndCutoff(const char* networkName, bool search)
{
	double cutoff = 1, quantileIndex = 6;
	if (search)
		searchCalibrations(cutoff, quantileIndex);
	else
	{
		for (int i = 0; i < gCalibrationTableSize; i++)
		{
			if (!strcmp(gCalibrationTable[i].networkName, networkName))
				cutoff = gCalibrationTable[i].cutoff, quantileIndex = gCalibrationTable[i].quantileIndex;
		}
		std::cout << " using preset cutoff " << cutoff << " and quantile index " << quantileIndex << std::endl;
	}
	return std::make_pair(cutoff, quantileFromIndex(quantileIndex));
}



#endif
