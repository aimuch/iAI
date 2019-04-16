#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <iomanip>

#include "NvInfer.h"
#include "common.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
using namespace nvinfer1;

static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;
static Logger gLogger;

const std::vector<std::string> directories{ "data/samples/mnist/", "data/mnist/" };
std::string locateFile(const std::string& input)
{
    return locateFile(input, directories);
}

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& fileName, uint8_t buffer[INPUT_H*INPUT_W])
{
    readPGMFile(fileName, buffer, INPUT_H, INPUT_W);
}

void onnxToTRTModel( const std::string& modelFile,        // name of the onnx model 
                     unsigned int maxBatchSize,            // batch size - NB must be at least as large as the batch we want to run with
                     IHostMemory *&trtModelStream)      // output buffer for the TensorRT model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    nvonnxparser::IOnnxConfig* config = nvonnxparser::createONNXConfig();
    config->setModelFileName(locateFile(modelFile, directories).c_str());
    
    nvonnxparser::IONNXParser* parser = nvonnxparser::createONNXParser(*config);
    
    //Optional - uncomment below lines to view network layer information
    //config->setPrintLayerInfo(true);
    //parser->reportParsingInfo();
    
    if (!parser->parse(locateFile(modelFile, directories).c_str(), DataType::kFLOAT))
    {
        string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    
    if (!parser->convertToTRTNetwork()) {
        string msg("ERROR, failed to convert onnx network into TRT network");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    nvinfer1::INetworkDefinition* network = parser->getTRTNetwork();
    
    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
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
    int inputIndex, outputIndex;
    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (engine.bindingIsInput(b))
            inputIndex = b;
        else
            outputIndex = b;
    }
    
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


int main(int argc, char** argv)
{
    // create a TensorRT model from the onnx model and serialize it to a stream
    IHostMemory *trtModelStream{nullptr};
    onnxToTRTModel("mnist.onnx", 1, trtModelStream);
    assert(trtModelStream != nullptr);

    // read a random digit file
    srand(unsigned(time(nullptr)));
    uint8_t fileData[INPUT_H*INPUT_W];
    int num = rand() % 10;
    readPGMFile(locateFile(std::to_string(num) + ".pgm", directories), fileData);

    // print an ascii representation
    std::cout << "\n\n\n---------------------------" << "\n\n\n" << std::endl;
    for (int i = 0; i < INPUT_H*INPUT_W; i++)
        std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");


    float data[INPUT_H*INPUT_W];
    for (int i = 0; i < INPUT_H*INPUT_W; i++)
        data[i] = 1.0 - float(fileData[i]/255.0);

    // deserialize the engine 
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // run inference
    float prob[OUTPUT_SIZE];
    doInference(*context, data, prob, 1);

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    std::cout << "\n\n";
    float val{0.0f};
    int idx{0};

    //Calculate Softmax
    float sum{0.0f};
    for(int i = 0; i < OUTPUT_SIZE; i++)
    {
        prob[i] = exp(prob[i]);
        sum += prob[i];
    }
    for(int i = 0; i < OUTPUT_SIZE; i++)
    {
        prob[i] /= sum;
        val = std::max(val, prob[i]);
        if (val == prob[i]) idx = i;
        
        cout << " Prob " << i << "  "<< std::fixed << std::setw(5) << std::setprecision(4) << prob[i];
        std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << std::endl;
    }
    std::cout << std::endl;

    return (idx == num && val > 0.9f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
