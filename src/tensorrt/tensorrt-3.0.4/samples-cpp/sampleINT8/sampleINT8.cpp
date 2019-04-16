#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <algorithm>
#include <float.h>
#include <string.h>
#include <chrono>
#include <iterator>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"

#include "BatchStream.h"
#include "LegacyCalibrator.h"


using namespace nvinfer1;
using namespace nvcaffeparser1;

static Logger gLogger;

// stuff we know about the network and the caffe input/output blobs

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const char* gNetworkName{nullptr};

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs;
    dirs.push_back(std::string("data/int8/") + gNetworkName + std::string("/"));
    dirs.push_back(std::string("data/") + gNetworkName + std::string("/"));
    return locateFile(input, dirs);
}

bool caffeToGIEModel(const std::string& deployFile,						// name for caffe prototxt
	const std::string& modelFile,						// name for model
	const std::vector<std::string>& outputs,			// network outputs
	unsigned int maxBatchSize,							// batch size - NB must be at least as large as the batch we want to run with)
	DataType dataType,
	IInt8Calibrator* calibrator,
	nvinfer1::IHostMemory *&gieModelStream)
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();

	if((dataType == DataType::kINT8 && !builder->platformHasFastInt8()) || (dataType == DataType::kHALF && !builder->platformHasFastFp16()))
		return false;
	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
		locateFile(modelFile).c_str(),
		*network,
		dataType == DataType::kINT8 ? DataType::kFLOAT : dataType);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 30);
	builder->setAverageFindIterations(1);
	builder->setMinFindIterations(1);
	builder->setDebugSync(true);
	builder->setInt8Mode(dataType == DataType::kINT8);
	builder->setHalf2Mode(dataType == DataType::kHALF);
	builder->setInt8Calibrator(calibrator);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	return true;
}

float doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];
	float ms{ 0.0f };

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	DimsCHW inputDims = static_cast<DimsCHW&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(INPUT_BLOB_NAME)));
	DimsCHW outputDims = static_cast<DimsCHW&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(OUTPUT_BLOB_NAME)));

	size_t inputSize = batchSize*inputDims.c()*inputDims.h()*inputDims.w() * sizeof(float), outputSize = batchSize * outputDims.c() * outputDims.h() * outputDims.w() * sizeof(float);
	CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
	CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

	CHECK(cudaMemcpy(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	cudaEvent_t start, end;
	CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
	CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
	cudaEventRecord(start, stream);
	context.enqueue(batchSize, buffers, stream, nullptr);
	cudaEventRecord(end, stream);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&ms, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	CHECK(cudaMemcpy(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	CHECK(cudaStreamDestroy(stream));
	return ms;
}


int calculateScore(float* batchProb, float* labels, int batchSize, int outputSize, int threshold)
{
	int success = 0;
	for (int i = 0; i < batchSize; i++)
	{
		float* prob = batchProb + outputSize*i, correct = prob[(int)labels[i]];

		int better = 0;
		for (int j = 0; j < outputSize; j++)
			if (prob[j] >= correct)
				better++;
		if (better <= threshold)
			success++;
	}
	return success;
}



class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
public:
	Int8EntropyCalibrator(BatchStream& stream, int firstBatch, bool readCache = true)
		: mStream(stream), mReadCache(readCache)
	{
		DimsNCHW dims = mStream.getDims();
		mInputCount = mStream.getBatchSize() * dims.c() * dims.h() * dims.w();
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
		std::ifstream input(calibrationTableName(), std::ios::binary);
		input >> std::noskipws;
		if (mReadCache && input.good())
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

		length = mCalibrationCache.size();
		return length ? &mCalibrationCache[0] : nullptr;
	}

	void writeCalibrationCache(const void* cache, size_t length) override
	{
		std::ofstream output(calibrationTableName(), std::ios::binary);
		output.write(reinterpret_cast<const char*>(cache), length);
	}

private:
    static std::string calibrationTableName()
    {
        assert(gNetworkName);
        return std::string("CalibrationTable") + gNetworkName;
    }
	BatchStream mStream;
	bool mReadCache{ true };

	size_t mInputCount;
	void* mDeviceInput{ nullptr };
	std::vector<char> mCalibrationCache;
};

std::pair<float, float> scoreModel(int batchSize, int firstBatch, int nbScoreBatches, DataType datatype, IInt8Calibrator* calibrator, bool quiet = false)
{
	IHostMemory *gieModelStream{ nullptr };
	bool valid = false;
    if (gNetworkName == std::string("mnist"))
        valid = caffeToGIEModel("deploy.prototxt", "mnist_lenet.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, batchSize, datatype, calibrator, gieModelStream);
    else
        valid = caffeToGIEModel("deploy.prototxt", std::string(gNetworkName) + ".caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, batchSize, datatype, calibrator, gieModelStream);

	if(!valid)
	{
		std::cout << "Engine could not be created at this precision" << std::endl;
		return std::pair<float, float>(0,0);
	}

	// Create engine and deserialize model.
	IRuntime* infer = createInferRuntime(gLogger);
	ICudaEngine* engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
	if (gieModelStream) gieModelStream->destroy();
	IExecutionContext* context = engine->createExecutionContext();

	BatchStream stream(batchSize, nbScoreBatches);
	stream.skip(firstBatch);

	DimsCHW outputDims = static_cast<DimsCHW&&>(context->getEngine().getBindingDimensions(context->getEngine().getBindingIndex(OUTPUT_BLOB_NAME)));
	int outputSize = outputDims.c()*outputDims.h()*outputDims.w();
	int top1{ 0 }, top5{ 0 };
	float totalTime{ 0.0f };
	std::vector<float> prob(batchSize * outputSize, 0);

	while (stream.next())
	{
		totalTime += doInference(*context, stream.getBatch(), &prob[0], batchSize);

		top1 += calculateScore(&prob[0], stream.getLabels(), batchSize, outputSize, 1);
		top5 += calculateScore(&prob[0], stream.getLabels(), batchSize, outputSize, 5);

		std::cout << (!quiet && stream.getBatchesRead() % 10 == 0 ? "." : "") << (!quiet && stream.getBatchesRead() % 800 == 0 ? "\n" : "") << std::flush;
	}
	int imagesRead = stream.getBatchesRead()*batchSize;
	float t1 = float(top1) / float(imagesRead), t5 = float(top5) / float(imagesRead);

	if (!quiet)
	{
		std::cout << "\nTop1: " << t1 << ", Top5: " << t5 << std::endl;
		std::cout << "Processing " << imagesRead << " images averaged " << totalTime / imagesRead << " ms/image and " << totalTime / stream.getBatchesRead() << " ms/batch." << std::endl;
	}

	context->destroy();
	engine->destroy();
	infer->destroy();
	return std::make_pair(t1, t5);
}




int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "Please provide the network as the first argument." << std::endl;
		exit(0);
	}
	gNetworkName = argv[1];

	int batchSize = 100, firstScoreBatch = 100, nbScoreBatches = 400;	// by default we score over 40K images starting at 10000, so we don't score those used to search calibration
	bool search = false;
	CalibrationAlgoType calibrationAlgo = CalibrationAlgoType::kENTROPY_CALIBRATION;

	for (int i = 2; i < argc; i++)
	{
		if (!strncmp(argv[i], "batch=", 6))
			batchSize = atoi(argv[i] + 6);
		else if (!strncmp(argv[i], "start=", 6))
			firstScoreBatch = atoi(argv[i] + 6);
		else if (!strncmp(argv[i], "score=", 6))
			nbScoreBatches = atoi(argv[i] + 6);
		else if (!strncmp(argv[i], "search", 6))
			search = true;
		else if (!strncmp(argv[i], "legacy", 6))
			calibrationAlgo = CalibrationAlgoType::kLEGACY_CALIBRATION;
		else
		{
			std::cout << "Unrecognized argument " << argv[i] << std::endl;
			exit(0);
		}
	}

	if (calibrationAlgo == CalibrationAlgoType::kENTROPY_CALIBRATION)
	{
		search = false;
	}

	if (batchSize > 128)
	{
		std::cout << "Please provide batch size <= 128" << std::endl;
		exit(0);
	}

	if ((firstScoreBatch + nbScoreBatches)*batchSize > 500000)
	{
		std::cout << "Only 50000 images available" << std::endl;
		exit(0);
	}

	std::cout.precision(6);


	BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);

	std::cout << "\nFP32 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
	scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kFLOAT, nullptr);

	std::cout << "\nFP16 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
	scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kHALF, nullptr);

	std::cout << "\nINT8 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
	if (calibrationAlgo == CalibrationAlgoType::kENTROPY_CALIBRATION)
	{
		Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH);
		scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kINT8, &calibrator);
	}
	else
	{
		std::pair<double, double> parameters = getQuantileAndCutoff(gNetworkName, search);
		Int8LegacyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH, parameters.first, parameters.second);
		scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kINT8, &calibrator);
	}

	shutdownProtobufLibrary();
	return 0;
}
