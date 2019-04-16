//! This sample builds a TensorRT engine by importing a trained MNIST Caffe model.
//! It uses the engine to run inference on an input image of a digit.

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "common.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

static Logger gLogger;

// Attributes of MNIST Caffe model
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const std::vector<std::string> directories{"data/samples/mnist/", "data/mnist/"};

std::string locateFile(const std::string& input)
{
    return locateFile(input, directories);
}

// Simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& fileName, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(fileName, buffer, INPUT_H, INPUT_W);
}

void caffeToTRTModel(const std::string& deployFile,           // Path of Caffe prototxt file
                     const std::string& modelFile,            // Path of Caffe model file
                     const std::vector<std::string>& outputs, // Names of network outputs
                     unsigned int maxBatchSize,               // Note: Must be at least as large as the batch we want to run with
                     IHostMemory*& trtModelStream)            // Output buffer for the TRT model
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Parse caffe model to populate network, then set the outputs
    const std::string deployFpath = locateFile(deployFile, directories);
    const std::string modelFpath = locateFile(modelFile, directories);
    std::cout << "Reading Caffe prototxt: " << deployFpath << "\n";
    std::cout << "Reading Caffe model: " << modelFpath << "\n";
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFpath.c_str(),
                                                              modelFpath.c_str(),
                                                              *network,
                                                              DataType::kFLOAT);

    // Specify output tensors of network
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);

    // Build engine
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // Destroy parser and network
    network->destroy();
    parser->destroy();

    // Serialize engine and destroy it
    trtModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();

    shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc > 1)
    {
        std::cout << "This sample builds a TensorRT engine by importing a trained MNIST Caffe model.\n";
        std::cout << "It uses the engine to run inference on an input image of a digit.\n";
        return EXIT_SUCCESS;
    }

    // Create TRT model from caffe model and serialize it to a stream
    IHostMemory* trtModelStream{nullptr};
    caffeToTRTModel("mnist.prototxt", "mnist.caffemodel", std::vector<std::string>{OUTPUT_BLOB_NAME}, 1, trtModelStream);
    assert(trtModelStream != nullptr);

    // Read a random digit file
    srand(unsigned(time(nullptr)));
    uint8_t fileData[INPUT_H * INPUT_W];
    const int num = rand() % 10;
    readPGMFile(locateFile(std::to_string(num) + ".pgm", directories), fileData);

    // Print ASCII representation of digit
    std::cout << "\nInput:\n" << std::endl;
    for (int i = 0; i < INPUT_H * INPUT_W; i++)
        std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

    // Parse mean file
    ICaffeParser* parser = createCaffeParser();
    IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mnist_mean.binaryproto", directories).c_str());
    parser->destroy();

    // Subtract mean from image
    const float* meanData = reinterpret_cast<const float*>(meanBlob->getData());

    float data[INPUT_H * INPUT_W];
    for (int i = 0; i < INPUT_H * INPUT_W; i++)
        data[i] = float(fileData[i]) - meanData[i];

    meanBlob->destroy();

    // Deserialize engine we serialized earlier
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference on input data
    float prob[OUTPUT_SIZE];
    doInference(*context, data, prob, 1);

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    float val{0.0f};
    int idx{0};
    for (unsigned int i = 0; i < 10; i++)
    {
        val = std::max(val, prob[i]);
        if (val == prob[i]) idx = i;
        std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
    std::cout << std::endl;

    return (idx == num && val > 0.9f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
