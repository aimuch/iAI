#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <string.h>
#include <map>
#include <random>
#include <iterator>

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

struct Params
{
	std::string deployFile, modelFile, engine, calibrationCache{"CalibrationTable"};
	std::vector<std::string> outputs;
	int device{ 0 }, batchSize{ 1 }, workspaceSize{ 16 }, iterations{ 10 }, avgRuns{ 10 };
	bool half2{ false }, int8{ false }, verbose{ false }, hostTime{ false };
} gParams;

static inline int volume(DimsCHW dims)
{
	return dims.c()*dims.h()*dims.w();
}

std::vector<std::string> gInputs;
std::map<std::string, DimsCHW> gInputDimensions;

// Logger for GIE info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO || gParams.verbose)
			std::cout << msg << std::endl;
	}
} gLogger;

class RndInt8Calibrator : public IInt8EntropyCalibrator
{
public:
	RndInt8Calibrator(int totalSamples, std::string cacheFile)
		: mTotalSamples(totalSamples)
		, mCurrentSample(0)
        , mCacheFile(cacheFile)
	{
		std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
		for(auto& elem: gInputDimensions)
		{
			int elemCount = volume(elem.second);

			std::vector<float> rnd_data(elemCount);
			for(auto& val: rnd_data)
				val = distribution(generator);

			void * data;
			CHECK(cudaMalloc(&data, elemCount * sizeof(float)));
			CHECK(cudaMemcpy(data, &rnd_data[0], elemCount * sizeof(float), cudaMemcpyHostToDevice));

			mInputDeviceBuffers.insert(std::make_pair(elem.first, data));
		}
	}

	~RndInt8Calibrator()
	{
		for(auto& elem: mInputDeviceBuffers)
			CHECK(cudaFree(elem.second));
	}

	int getBatchSize() const override
	{
		return 1;
	}
	
	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
		if (mCurrentSample >= mTotalSamples)
			return false;

		for(int i = 0; i < nbBindings; ++i)
			bindings[i] = mInputDeviceBuffers[names[i]];

		++mCurrentSample;
		return true;
	}

	const void* readCalibrationCache(size_t& length) override
	{
        mCalibrationCache.clear();
		std::ifstream input(mCacheFile, std::ios::binary);
		input >> std::noskipws;
		if (input.good())
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

		length = mCalibrationCache.size();
		return length ? &mCalibrationCache[0] : nullptr;
	}

	virtual void writeCalibrationCache(const void* cache, size_t length) override
	{
	}

private:
	int mTotalSamples;
	int mCurrentSample;
    std::string mCacheFile;
	std::map<std::string, void*> mInputDeviceBuffers;
	std::vector<char> mCalibrationCache;
};

ICudaEngine* caffeToGIEModel()
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(gParams.deployFile.c_str(),
															  gParams.modelFile.empty() ? 0 : gParams.modelFile.c_str(),
															  *network,
															  gParams.half2 ? DataType::kHALF:DataType::kFLOAT);


	if (!blobNameToTensor)
		return nullptr;

	for (int i = 0, n = network->getNbInputs(); i < n; i++)
	{
		DimsCHW dims = static_cast<DimsCHW&&>(network->getInput(i)->getDimensions());
		gInputs.push_back(network->getInput(i)->getName());
		gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
		std::cout << "Input \"" << network->getInput(i)->getName() << "\": " << dims.c() << "x" << dims.h() << "x" << dims.w() << std::endl;
	}

	// specify which tensors are outputs
	for (auto& s : gParams.outputs)
	{
		if (blobNameToTensor->find(s.c_str()) == nullptr)
		{
			std::cout << "could not find output blob " << s << std::endl;
			return nullptr;
		}
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	}

	for (int i = 0, n = network->getNbOutputs(); i < n; i++)
	{
		DimsCHW dims = static_cast<DimsCHW&&>(network->getOutput(i)->getDimensions());
		std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.c() << "x" << dims.h() << "x" << dims.w() << std::endl;
	}

	// Build the engine
	builder->setMaxBatchSize(gParams.batchSize);
	builder->setMaxWorkspaceSize(size_t(gParams.workspaceSize)<<20);
	builder->setHalf2Mode(gParams.half2);

	RndInt8Calibrator calibrator(1, gParams.calibrationCache);
	if (gParams.int8)
	{
		builder->setInt8Mode(true);
		builder->setInt8Calibrator(&calibrator);
	}

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	if (engine == nullptr)
		std::cout << "could not build engine" << std::endl;

	parser->destroy();
	network->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
	return engine;
}

void createMemory(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
	size_t bindingIndex = engine.getBindingIndex(name.c_str());
	printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
	assert(bindingIndex < buffers.size());
	DimsCHW dimensions = static_cast<DimsCHW&&>(engine.getBindingDimensions((int)bindingIndex));
	size_t eltCount = dimensions.c()*dimensions.h()*dimensions.w()*gParams.batchSize, memSize = eltCount * sizeof(float);

	float* localMem = new float[eltCount];
	for (size_t i = 0; i < eltCount; i++)
		localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;

	void* deviceMem;
	CHECK(cudaMalloc(&deviceMem, memSize));
	if (deviceMem == nullptr)
	{
		std::cerr << "Out of memory" << std::endl;
		exit(1);
	}
	CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));

	delete[] localMem;
	buffers[bindingIndex] = deviceMem;	
}

void doInference(ICudaEngine& engine)
{
	IExecutionContext *context = engine.createExecutionContext();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.

	std::vector<void*> buffers(gInputs.size() + gParams.outputs.size());
	for (size_t i = 0; i < gInputs.size(); i++)
		createMemory(engine, buffers, gInputs[i]);

	for (size_t i = 0; i < gParams.outputs.size(); i++)
		createMemory(engine, buffers, gParams.outputs[i]);

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	cudaEvent_t start, end;
	CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
	CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

	for (int j = 0; j < gParams.iterations; j++)
	{
		float total = 0, ms;
		for (int i = 0; i < gParams.avgRuns; i++)
		{
			if (gParams.hostTime)
			{
				auto t_start = std::chrono::high_resolution_clock::now();
				context->execute(gParams.batchSize, &buffers[0]);
				auto t_end = std::chrono::high_resolution_clock::now();
				ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
			}
			else
			{
				cudaEventRecord(start, stream);
				context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
				cudaEventRecord(end, stream);
				cudaEventSynchronize(end);
				cudaEventElapsedTime(&ms, start, end);
			}
			total += ms;
		}
		total /= gParams.avgRuns;
		std::cout << "Average over " << gParams.avgRuns << " runs is " << total << " ms." << std::endl;
	}


	cudaStreamDestroy(stream);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}



static void printUsage()
{
	printf("\n");
	printf("Mandatory params:\n");
	printf("  --deploy=<file>      Caffe deploy file\n");
	printf("  --output=<name>      Output blob name (can be specified multiple times)\n");

	printf("\nOptional params:\n");

	printf("  --model=<file>       Caffe model file (default = no model, random weights used)\n");
	printf("  --batch=N            Set batch size (default = %d)\n", gParams.batchSize);
	printf("  --device=N           Set cuda device to N (default = %d)\n", gParams.device);
	printf("  --iterations=N       Run N iterations (default = %d)\n", gParams.iterations);
	printf("  --avgRuns=N          Set avgRuns to N - perf is measured as an average of avgRuns (default=%d)\n", gParams.avgRuns);
	printf("  --workspace=N        Set workspace size in megabytes (default = %d)\n", gParams.workspaceSize);
	printf("  --half2              Run in paired fp16 mode (default = false)\n");
	printf("  --int8               Run in int8 mode (default = false)\n");
	printf("  --verbose            Use verbose logging (default = false)\n");
	printf("  --hostTime	       Measure host time rather than GPU time (default = false)\n");
	printf("  --engine=<file>      Generate a serialized GIE engine\n");
	printf("  --calib=<file>       Read INT8 calibration cache file\n");

	fflush(stdout);
}

bool parseString(const char* arg, const char* name, std::string& value)
{
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
	if (match)
	{
		value = arg + n + 3;
		std::cout << name << ": " << value << std::endl;
	}
	return match;
}

bool parseInt(const char* arg, const char* name, int& value)
{
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
	if (match)
	{
		value = atoi(arg + n + 3);
		std::cout << name << ": " << value << std::endl;
	}
	return match;
}

bool parseBool(const char* arg, const char* name, bool& value)
{
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n);
	if (match)
	{
		std::cout << name << std::endl;
		value = true;
	}
	return match;

}


bool parseArgs(int argc, char* argv[])
{
	if (argc < 3)
	{
		printUsage();
		return false;
	}

	for (int j = 1; j < argc; j++)
	{
		if (parseString(argv[j], "model", gParams.modelFile) || parseString(argv[j], "deploy", gParams.deployFile) || parseString(argv[j], "engine", gParams.engine))
			continue;

		if (parseString(argv[j], "calib", gParams.calibrationCache))
			continue;
		
		std::string output;
		if (parseString(argv[j], "output", output))
		{
			gParams.outputs.push_back(output);
			continue;
		}

		if (parseInt(argv[j], "batch", gParams.batchSize) || parseInt(argv[j], "iterations", gParams.iterations) || parseInt(argv[j], "avgRuns", gParams.avgRuns) 
			|| parseInt(argv[j], "device", gParams.device)	|| parseInt(argv[j], "workspace", gParams.workspaceSize))
			continue;

		if (parseBool(argv[j], "half2", gParams.half2) || parseBool(argv[j], "int8", gParams.int8)
			|| parseBool(argv[j], "verbose", gParams.verbose) || parseBool(argv[j], "hostTime", gParams.hostTime))
			continue;

		printf("Unknown argument: %s\n", argv[j]);
		return false;
	}
	return true;
}

static ICudaEngine* createEngine()
{
	ICudaEngine *engine;

	if (!gParams.deployFile.empty()) {
		engine = caffeToGIEModel();
		if (!engine)
		{
			std::cerr << "Engine could not be created" << std::endl;
			return nullptr;
		}
	
	
		if (!gParams.engine.empty())
		{
			std::ofstream p(gParams.engine);
			if (!p)
			{
				std::cerr << "could not open plan output file" << std::endl;
				return nullptr;
			}
			IHostMemory *ptr = engine->serialize();
            assert(ptr);
            p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
            ptr->destroy();
		}
		return engine;
	}

	// load directly from serialized engine file if deploy not specified
	if (!gParams.engine.empty()) {
		char *gieModelStream{nullptr};
        size_t size{0};
		std::ifstream file(gParams.engine, std::ios::binary);
		if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            gieModelStream = new char[size];
            assert(gieModelStream);
            file.read(gieModelStream, size);
			file.close();
		}

		IRuntime* infer = createInferRuntime(gLogger);
		engine = infer->deserializeCudaEngine(gieModelStream, size, nullptr);
        if (gieModelStream) delete [] gieModelStream;

		// assume input to be "data" for deserialized engine
		gInputs.push_back("data");
		return engine;
	}

	// complain about empty deploy file
	std::cerr << "Deploy file not specified" << std::endl;
	return nullptr;
}

int main(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream

	if (!parseArgs(argc, argv))
		return -1;

	cudaSetDevice(gParams.device);

	if (gParams.outputs.size() == 0)
	{
		std::cerr << "At least one network output must be defined" << std::endl;
		return -1;
	}

	ICudaEngine* engine = createEngine();
	if (!engine)
	{
		std::cerr << "Engine could not be created" << std::endl;
		return -1;
	}

	doInference(*engine);
	engine->destroy();

	return 0;
}
