#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <string.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;
static Logger gLogger;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/mnist/", "data/mnist/"};
    return locateFile(input, dirs);
}

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& filename,  uint8_t buffer[INPUT_H*INPUT_W])
{
    readPGMFile(locateFile(filename), buffer, INPUT_H, INPUT_W);
}

void caffeToGIEModel(const std::string& deployFile,					// name for caffe prototxt
					 const std::string& modelFile,					// name for model 
					 const std::vector<std::string>& outputs,		// network outputs
					 unsigned int maxBatchSize,						// batch size - NB must be at least as large as the batch we want to run with)
					 nvcaffeparser1::IPluginFactory* pluginFactory,	// factory for plugin layers
					 IHostMemory *&gieModelStream)					// output stream for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactory(pluginFactory);

	bool fp16 = builder->platformHasFastFp16();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
															  locateFile(modelFile).c_str(),
															  *network,
															  fp16 ? DataType::kHALF : DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);
	builder->setHalf2Mode(fp16);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();

	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}


class FCPlugin: public IPlugin
{
public:
	FCPlugin(const Weights *weights, int nbWeights, int nbOutputChannels): mNbOutputChannels(nbOutputChannels)
	{
		// since we want to deal with the case where there is no bias, we can't infer
		// the number of channels from the bias weights.

		assert(nbWeights == 2);
		mKernelWeights = copyToDevice(weights[0].values, weights[0].count);
		mBiasWeights = copyToDevice(weights[1].values, weights[1].count);
		assert(mBiasWeights.count == 0 || mBiasWeights.count == nbOutputChannels);

		mNbInputChannels = int(weights[0].count / nbOutputChannels);
	}

	// create the plugin at runtime from a byte stream
	FCPlugin(const void* data, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(data), *a = d;
		mNbInputChannels = read<int>(d);
		mNbOutputChannels = read<int>(d);
		int biasCount = read<int>(d);

		mKernelWeights = deserializeToDevice(d, mNbInputChannels * mNbOutputChannels);
		mBiasWeights = deserializeToDevice(d, biasCount);
		assert(d == a + length);
	}

	~FCPlugin()
	{
		cudaFree(const_cast<void*>(mKernelWeights.values));
		cudaFree(const_cast<void*>(mBiasWeights.values));
	}

	int getNbOutputs() const override
	{
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
		assert(mNbInputChannels == inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2]);
		return DimsCHW(mNbOutputChannels, 1, 1);
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
	{
	}

	int initialize() override
	{
		CHECK(cudnnCreate(&mCudnn));							// initialize cudnn and cublas
		CHECK(cublasCreate(&mCublas));
		CHECK(cudnnCreateTensorDescriptor(&mSrcDescriptor));	// create cudnn tensor descriptors we need for bias addition
		CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));

		return 0;
	}

	virtual void terminate() override
	{
		CHECK(cublasDestroy(mCublas));
		CHECK(cudnnDestroy(mCudnn));
	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
	{
		float kONE = 1.0f, kZERO = 0.0f;
		cublasSetStream(mCublas, stream);
		cudnnSetStream(mCudnn, stream);
		CHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mNbOutputChannels, batchSize, mNbInputChannels, &kONE, 
				reinterpret_cast<const float*>(mKernelWeights.values), mNbInputChannels, 
				reinterpret_cast<const float*>(inputs[0]), mNbInputChannels, &kZERO, 
				reinterpret_cast<float*>(outputs[0]), mNbOutputChannels));
		if (mBiasWeights.count)
		{
			CHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, mNbOutputChannels, 1, 1));
			CHECK(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, mNbOutputChannels, 1, 1));
			CHECK(cudnnAddTensor(mCudnn, &kONE, mSrcDescriptor, mBiasWeights.values, &kONE, mDstDescriptor, outputs[0]));
		}
		return 0;
	}

	virtual size_t getSerializationSize() override
	{
		// 3 integers (number of input channels, number of output channels, bias size), and then the weights:
		return sizeof(int)*3 + mKernelWeights.count*sizeof(float) + mBiasWeights.count*sizeof(float);
	}

	virtual void serialize(void* buffer) override
	{
		char* d = reinterpret_cast<char*>(buffer), *a = d;

		write(d, mNbInputChannels);
		write(d, mNbOutputChannels);
		write(d, (int)mBiasWeights.count);
		serializeFromDevice(d, mKernelWeights);
		serializeFromDevice(d, mBiasWeights);

		assert(d == a + getSerializationSize());
	}
private:
	template<typename T> void write(char*& buffer, const T& val)
	{
		*reinterpret_cast<T*>(buffer) = val;
		buffer += sizeof(T);
	}

	template<typename T> T read(const char*& buffer)
	{
		T val = *reinterpret_cast<const T*>(buffer);
		buffer += sizeof(T);
		return val;
	}

	Weights copyToDevice(const void* hostData, size_t count)
	{
		void* deviceData;
		CHECK(cudaMalloc(&deviceData, count * sizeof(float)));
		CHECK(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
		return Weights{ DataType::kFLOAT, deviceData, int64_t(count) };
	}

	void serializeFromDevice(char*& hostBuffer, Weights deviceWeights)
	{		
		cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
		hostBuffer += deviceWeights.count * sizeof(float);
	}

	Weights deserializeToDevice(const char*& hostBuffer, size_t count)
	{
		Weights w = copyToDevice(hostBuffer, count);
		hostBuffer += count * sizeof(float);
		return w;	
	}

	int mNbOutputChannels, mNbInputChannels;
	cudnnHandle_t mCudnn;
	cublasHandle_t mCublas;
	Weights mKernelWeights, mBiasWeights;
	cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
};

// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
	// caffe parser plugin implementation
	bool isPlugin(const char* name) override
	{
		return !strcmp(name, "ip2");
	}

	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
		// there's no way to pass parameters through from the model definition, so we have to define it here explicitly
		static const int NB_OUTPUT_CHANNELS = 10;	
		assert(isPlugin(layerName) && nbWeights == 2 && weights[0].type == DataType::kFLOAT && weights[1].type == DataType::kFLOAT);
		assert(mPlugin.get() == nullptr);
		mPlugin = std::unique_ptr<FCPlugin>(new FCPlugin(weights, nbWeights, NB_OUTPUT_CHANNELS));
		return mPlugin.get();
	}

	// deserialization plugin implementation
	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{		
		assert(isPlugin(layerName));
		assert(mPlugin.get() == nullptr);
		mPlugin = std::unique_ptr<FCPlugin>(new FCPlugin(serialData, serialLength));
		return mPlugin.get();
	}

	// the application has to destroy the plugin when it knows it's safe to do so
	void destroyPlugin()
	{
		mPlugin.release();
	}

	std::unique_ptr<FCPlugin> mPlugin{ nullptr };
};

int main(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream
	PluginFactory pluginFactory;
	IHostMemory *gieModelStream{ nullptr };
	caffeToGIEModel("mnist.prototxt", "mnist.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, &pluginFactory, gieModelStream);
	pluginFactory.destroyPlugin();
	
	// read a random digit file
	srand(unsigned(time(nullptr)));
	uint8_t fileData[INPUT_H*INPUT_W];
    int num{rand()%10};
	readPGMFile(std::to_string(num) + ".pgm", fileData);

	// print an ascii representation
	std::cout << "\n\n\n---------------------------" << "\n\n\n" << std::endl;
	for (int i = 0; i < INPUT_H*INPUT_W; i++)
		std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

	ICaffeParser* parser = createCaffeParser();
	IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mnist_mean.binaryproto").c_str());
	parser->destroy();

	// parse the mean file and 	subtract it from the image
	const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());

	float data[INPUT_H*INPUT_W];
	for (int i = 0; i < INPUT_H*INPUT_W; i++)
		data[i] = float(fileData[i])-meanData[i];

	meanBlob->destroy();

	// deserialize the engine 
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);

	IExecutionContext *context = engine->createExecutionContext();

	// run inference
	float prob[OUTPUT_SIZE];
	doInference(*context, data, prob, 1);

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	pluginFactory.destroyPlugin();

	// print a histogram of the output distribution
	std::cout << "\n\n";
    bool pass{false};
	for (int i = 0; i < 10; i++)
    {
        int res = std::floor(prob[i] * 10 + 0.5);
        if (res == 10 && i == num) pass = true;
		std::cout << i << ": " << std::string(res, '*') << "\n";
    }
	std::cout << std::endl;

	return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
