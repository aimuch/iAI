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
#include <cstdint>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "fp16.h"
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

void caffeToTRTModel(const std::string& deployFile,                 // name for caffe prototxt
                     const std::string& modelFile,                  // name for model
                     const std::vector<std::string>& outputs,       // network outputs
                     unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
                     nvcaffeparser1::IPluginFactoryExt* pluginFactory, // factory for plugin layers
                     IHostMemory *&trtModelStream)                  // output stream for the TensorRT model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactoryExt(pluginFactory);

    bool fp16 = builder->platformHasFastFp16();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
                                                              locateFile(modelFile).c_str(),
                                                              *network, fp16 ? DataType::kHALF : DataType::kFLOAT);

    // specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    builder->setFp16Mode(fp16);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();

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


class FCPlugin: public IPluginExt
{
public:
    FCPlugin(const Weights *weights, int nbWeights, int nbOutputChannels): mNbOutputChannels(nbOutputChannels)
    {
        assert(nbWeights == 2);

        mKernelWeights = weights[0];
        assert(mKernelWeights.type == DataType::kFLOAT || mKernelWeights.type == DataType::kHALF);

        mBiasWeights = weights[1];
        assert(mBiasWeights.count == 0 || mBiasWeights.count == nbOutputChannels);
        assert(mBiasWeights.type == DataType::kFLOAT || mBiasWeights.type == DataType::kHALF);

        mKernelWeights.values = malloc(mKernelWeights.count*type2size(mKernelWeights.type));
        memcpy(const_cast<void*>(mKernelWeights.values), weights[0].values, mKernelWeights.count*type2size(mKernelWeights.type));
        mBiasWeights.values = malloc(mBiasWeights.count*type2size(mBiasWeights.type));
        memcpy(const_cast<void*>(mBiasWeights.values), weights[1].values, mBiasWeights.count*type2size(mBiasWeights.type));

        mNbInputChannels = int(weights[0].count / nbOutputChannels);
    }

    // create the plugin at runtime from a byte stream
    FCPlugin(const void* data, size_t length)
    {
        const char* d = static_cast<const char*>(data), *a = d;
        read(d, mNbInputChannels);
        read(d, mNbOutputChannels);

        mKernelWeights.count = mNbInputChannels * mNbOutputChannels;
        mKernelWeights.values = nullptr;

        read(d, mBiasWeights.count);
        mBiasWeights.values = nullptr;

        read(d, mDataType);

        deserializeToDevice(d, mDeviceKernel, mKernelWeights.count*type2size(mDataType));
        deserializeToDevice(d, mDeviceBias, mBiasWeights.count*type2size(mDataType));
        assert(d == a + length);
    }

    ~FCPlugin()
    {
        if (mKernelWeights.values)
        {
            free(const_cast<void*>(mKernelWeights.values));
            mKernelWeights.values = nullptr;
        }
        if (mBiasWeights.values)
        {
            free(const_cast<void*>(mBiasWeights.values));
            mBiasWeights.values = nullptr;
        }
    }

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
        assert(mNbInputChannels == inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2]);
        return Dims3(mNbOutputChannels, 1, 1);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override { return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
        mDataType = type;
    }

    int initialize() override
    {
        CHECK(cudnnCreate(&mCudnn));// initialize cudnn and cublas
        CHECK(cublasCreate(&mCublas));
        CHECK(cudnnCreateTensorDescriptor(&mSrcDescriptor));// create cudnn tensor descriptors we need for bias addition
        CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));
        if (mKernelWeights.values)
            convertAndCopyToDevice(mDeviceKernel, mKernelWeights);
        if (mBiasWeights.values)
            convertAndCopyToDevice(mDeviceBias, mBiasWeights);

        return 0;
    }

    virtual void terminate() override
    {
        CHECK(cudnnDestroyTensorDescriptor(mSrcDescriptor));
        CHECK(cudnnDestroyTensorDescriptor(mDstDescriptor));
        CHECK(cublasDestroy(mCublas));
        CHECK(cudnnDestroy(mCudnn));
        if (mDeviceKernel)
        {
            cudaFree(mDeviceKernel);
            mDeviceKernel = nullptr;
        }
        if (mDeviceBias)
        {
            cudaFree(mDeviceBias);
            mDeviceBias = nullptr;
        }
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }

    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        float onef{1.0f}, zerof{0.0f};
        __half oneh = fp16::__float2half(1.0f), zeroh = fp16::__float2half(0.0f);

        cublasSetStream(mCublas, stream);
        cudnnSetStream(mCudnn, stream);

        if (mDataType == DataType::kFLOAT)
        {
            CHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mNbOutputChannels, batchSize, mNbInputChannels, &onef,
                              reinterpret_cast<const float*>(mDeviceKernel), mNbInputChannels,
                              reinterpret_cast<const float*>(inputs[0]), mNbInputChannels, &zerof,
                              reinterpret_cast<float*>(outputs[0]), mNbOutputChannels));
        }
        else
        {
            CHECK(cublasHgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mNbOutputChannels, batchSize, mNbInputChannels, &oneh,
                              reinterpret_cast<const __half*>(mDeviceKernel), mNbInputChannels,
                              reinterpret_cast<const __half*>(inputs[0]), mNbInputChannels, &zeroh,
                              reinterpret_cast<__half*>(outputs[0]), mNbOutputChannels));
        }
        if (mBiasWeights.count)
        {
            cudnnDataType_t cudnnDT = mDataType == DataType::kFLOAT ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
            CHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, cudnnDT, 1, mNbOutputChannels, 1, 1));
            CHECK(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, cudnnDT, batchSize, mNbOutputChannels, 1, 1));
            CHECK(cudnnAddTensor(mCudnn, &onef, mSrcDescriptor, mDeviceBias, &onef, mDstDescriptor, outputs[0]));
        }

        return 0;
    }

    virtual size_t getSerializationSize() override
    {
        return sizeof(mNbInputChannels) + sizeof(mNbOutputChannels) + sizeof(mBiasWeights.count) + sizeof(mDataType) +
               (mKernelWeights.count + mBiasWeights.count) * type2size(mDataType);
    }

    virtual void serialize(void* buffer) override
    {
        char* d = static_cast<char*>(buffer), *a = d;

        write(d, mNbInputChannels);
        write(d, mNbOutputChannels);
        write(d, mBiasWeights.count);
        write(d, mDataType);
        convertAndCopyToBuffer(d, mKernelWeights);
        convertAndCopyToBuffer(d, mBiasWeights);
        assert(d == a + getSerializationSize());
    }

private:
    size_t type2size(DataType type) { return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); }

    template<typename T> void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }

    void* copyToDevice(const void* data, size_t count)
    {
        void* deviceData;
        CHECK(cudaMalloc(&deviceData, count));
        CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
        return deviceData;
    }

    void convertAndCopyToDevice(void*& deviceWeights, const Weights& weights)
    {
        if (weights.type != mDataType) // Weights are converted in host memory first, if the type does not match
        {
            size_t size = weights.count*(mDataType == DataType::kFLOAT ? sizeof(float) : sizeof(__half));
            void* buffer = malloc(size);
            for (int64_t v = 0; v < weights.count; ++v)
                if (mDataType == DataType::kFLOAT)
                    static_cast<float*>(buffer)[v] = fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
                else
                    static_cast<__half*>(buffer)[v] = fp16::__float2half(static_cast<const float*>(weights.values)[v]);

            deviceWeights = copyToDevice(buffer, size);
            free(buffer);
        }
        else
            deviceWeights = copyToDevice(weights.values, weights.count * type2size(mDataType));
    }

    void convertAndCopyToBuffer(char*& buffer, const Weights& weights)
    {
        if (weights.type != mDataType)
            for (int64_t v = 0; v < weights.count; ++v)
                if (mDataType == DataType::kFLOAT)
                    reinterpret_cast<float*>(buffer)[v] = fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
                else
                    reinterpret_cast<__half*>(buffer)[v] = fp16::__float2half(static_cast<const float*>(weights.values)[v]);
        else
            memcpy(buffer, weights.values, weights.count * type2size(mDataType));
        buffer += weights.count * type2size(mDataType);
    }

    void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
    {
        deviceWeights = copyToDevice(hostBuffer, size);
        hostBuffer += size;
    }

    int mNbOutputChannels, mNbInputChannels;
    Weights mKernelWeights, mBiasWeights;

    DataType mDataType{DataType::kFLOAT};
    void* mDeviceKernel{nullptr};
    void* mDeviceBias{nullptr};

    cudnnHandle_t mCudnn;
    cublasHandle_t mCublas;
    cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
};

// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryExt
{
public:
    // caffe parser plugin implementation
    bool isPlugin(const char* name) override
    {
        return isPluginExt(name);
    }

    bool isPluginExt(const char* name) override
    {
        return !strcmp(name, "ip2");
    }

    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
    {
        // there's no way to pass parameters through from the model definition, so we have to define it here explicitly
        static const int NB_OUTPUT_CHANNELS = 10;
        assert(isPlugin(layerName) && nbWeights == 2);
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

    // User application destroys plugin when it is safe to do so.
    // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
    void destroyPlugin()
    {
        mPlugin.reset();
    }

    std::unique_ptr<FCPlugin> mPlugin{ nullptr };
};

int main(int argc, char** argv)
{
    // create a TensorRT model from the caffe model and serialize it to a stream
    PluginFactory parserPluginFactory;
    IHostMemory *trtModelStream{ nullptr };
    caffeToTRTModel("mnist.prototxt", "mnist.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, &parserPluginFactory, trtModelStream);
    parserPluginFactory.destroyPlugin();
    assert(trtModelStream != nullptr);

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
    assert(parser != nullptr);
    IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mnist_mean.binaryproto").c_str());
    parser->destroy();

    // parse the mean file and     subtract it from the image
    const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());

    float data[INPUT_H*INPUT_W];
    for (int i = 0; i < INPUT_H*INPUT_W; i++)
        data[i] = float(fileData[i])-meanData[i];

    meanBlob->destroy();

    // deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    PluginFactory pluginFactory;
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), &pluginFactory);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // run inference
    float prob[OUTPUT_SIZE];
    doInference(*context, data, prob, 1);

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Destroy plugins created by factory
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
