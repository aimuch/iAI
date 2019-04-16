#include "NvInfer.h"
#include "common.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <iterator>
#include <vector>

// constants that are known about the MNIST MLP network.
static const int32_t INPUT_H{28};                                                    // The height of the mnist input image.
static const int32_t INPUT_W{28};                                                    // The weight of the mnist input image.
static const int32_t HIDDEN_COUNT{2};                                                // The number of hidden layers for MNIST sample.
static const int32_t HIDDEN_SIZE{256};                                               // The size of the hidden state for MNIST sample.
static const int32_t FINAL_SIZE{10};                                                 // The size of the output state for MNIST sample.
static const int32_t MAX_BATCH_SIZE{1};                                              // The maximum default batch size for MNIST sample.
static const int32_t OUTPUT_SIZE{10};                                                // The output of the final MLP layer for MNIST sample.
static const int32_t ITER_COUNT{1};                                                  // The number of iterations to run the MNIST sample.
static const nvinfer1::ActivationType MNIST_ACT{nvinfer1::ActivationType::kSIGMOID}; // The MNIST sample uses a sigmoid for activation.
static const char* INPUT_BLOB_NAME{"input"};                                         // The default input blob name.
static const char* OUTPUT_BLOB_NAME{"output"};                                       // the default output blob name.
static const char* DEFAULT_WEIGHT_FILE{"sampleMLP.wts2"};                            // The weight file produced from README.txt
static int gUseDLACore{-1};                                                          // The DLA core to run sample on.

static Logger gLogger;
/**
 * \class ShapedWeights
 * \brief A combination of Dims and Weights to provide shape to a weight struct.
 */
struct ShapedWeights
{
    nvinfer1::Dims shape;
    nvinfer1::Weights data;
};
typedef std::map<std::string, ShapedWeights> WeightMap_t;

// The split function takes string and based on a set of tokens produces a vector of tokens
// tokenized by the tokens. This is used to parse the shape field of the wts format.
static void split(std::vector<std::string>& split, std::string tokens, const std::string& input)
{
    split.clear();
    std::size_t begin = 0, size = input.size();
    while (begin != std::string::npos)
    {
        std::size_t found = input.find_first_of(tokens, begin);
        // Handle case of two or more delimiters in a row.
        if (found != begin)
            split.push_back(input.substr(begin, found - begin));
        begin = found + 1;
        // Handle case of no more tokens.
        if (found == std::string::npos)
            break;
        // Handle case of delimiter being last or first token.
        if (begin >= size)
            break;
    }
}

// Read a data blob from the input file.
void* loadShapeData(std::ifstream& input, size_t numElements)
{
    void* tmp = malloc(sizeof(float) * numElements);
    input.read(static_cast<char*>(tmp), numElements * sizeof(float));
    assert(input.peek() == '\n');
    // Consume the newline at the end of the data blob.
    input.get();
    return tmp;
}

nvinfer1::Dims loadShape(std::ifstream& input)
{
    // Initial format is "(A, B, C,...,Y [,])"
    nvinfer1::Dims shape{};
    std::string shapeStr;

    // Convert to "(A,B,C,...,Y[,])"
    do
    {
        std::string tmp;
        input >> tmp;
        shapeStr += tmp;
    } while (*shapeStr.rbegin() != ')');
    assert(input.peek() == ' ');

    // Consume the space between the shape and the data buffer.
    input.get();

    // Convert to "A,B,C,...,Y[,]"
    assert(*shapeStr.begin() == '(');
    shapeStr.erase(0, 1); //
    assert(*shapeStr.rbegin() == ')');
    shapeStr.pop_back();

    // Convert to "A,B,C,...,Y"
    if (*shapeStr.rbegin() == ',')
        shapeStr.pop_back(); // Remove the excess ',' character

    std::vector<std::string> shapeDim;
    split(shapeDim, ",", shapeStr);
    // Convert to {A, B, C,...,Y}
    assert(shapeDim.size() <= shape.MAX_DIMS);
    assert(shapeDim.size() > 0);
    assert(shape.nbDims == 0);
    std::for_each(shapeDim.begin(),
                  shapeDim.end(),
                  [&](std::string& val) {
                      shape.d[shape.nbDims++] = std::stoi(val);
                  });
    return shape;
}

// Our weight files are in a very simple space delimited format.
// type is the integer value of the DataType enum in NvInfer.h.
// <number of buffers>
// for each buffer: [name] [type] [size] <data x size in hex>
WeightMap_t loadWeights(const std::string file)
{
    WeightMap_t weightMap;
    std::ifstream input(file, std::ios_base::binary);
    assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while (count--)
    {
        ShapedWeights wt{};
        std::int32_t type;
        std::string name;
        input >> name >> std::dec >> type;
        wt.shape = loadShape(input);
        wt.data.type = static_cast<nvinfer1::DataType>(type);
        wt.data.count = std::accumulate(wt.shape.d, wt.shape.d + wt.shape.nbDims, 1, std::multiplies<int32_t>());
        assert(wt.data.type == nvinfer1::DataType::kFLOAT);
        wt.data.values = loadShapeData(input, wt.data.count);
        weightMap[name] = wt;
    }
    return weightMap;
}

// We have the data files located in a specific directory. This
// searches for that directory format from the current directory.
std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/mnist/", "data/mnist/",
                                  "data/samples/mlp/", "data/mlp/"};
    return locateFile(input, dirs);
}

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& filename, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(locateFile(filename), buffer, INPUT_H, INPUT_W);
}

// The addMLPLayer function is a simple helper function that creates the combination required for an
// MLP layer. By replacing the implementation of this sequence with various implementations, then
// then it can be shown how TensorRT optimizations those layer sequences.
nvinfer1::ILayer* addMLPLayer(nvinfer1::INetworkDefinition* network,
                              nvinfer1::ITensor& inputTensor,
                              int32_t hiddenSize,
                              nvinfer1::Weights wts,
                              nvinfer1::Weights bias,
                              nvinfer1::ActivationType actType,
                              int idx)
{
    std::string baseName("MLP Layer" + (idx == -1 ? "Output" : std::to_string(idx)));
    auto fc = network->addFullyConnected(inputTensor, hiddenSize, wts, bias);
    assert(fc != nullptr);
    std::string fcName = baseName + "FullyConnected";
    fc->setName(fcName.c_str());
    auto act = network->addActivation(*fc->getOutput(0), actType);
    assert(act != nullptr);
    std::string actName = baseName + "Activation";
    act->setName(actName.c_str());
    return act;
}

void transposeWeights(nvinfer1::Weights& wts, int hiddenSize)
{
    int d = 0;
    int dim0 = hiddenSize;       // 256 or 10
    int dim1 = wts.count / dim0; // 784 or 256
    uint32_t* trans_wts = new uint32_t[wts.count];
    for (int d0 = 0; d0 < dim0; ++d0)
    {
        for (int d1 = 0; d1 < dim1; ++d1)
        {
            trans_wts[d] = *((uint32_t*) wts.values + d1 * dim0 + d0);
            d++;
        }
    }

    for (int k = 0; k < wts.count; ++k)
    {
        *((uint32_t*) wts.values + k) = trans_wts[k];
    }
}

// Create the Engine using only the API and not any parser.
nvinfer1::ICudaEngine* fromAPIToModel(nvinfer1::IBuilder* builder)
{
    nvinfer1::DataType dt{nvinfer1::DataType::kFLOAT};
    WeightMap_t weightMap{loadWeights(locateFile(DEFAULT_WEIGHT_FILE))};
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    // FC layers must still have 3 dimensions, so we create a {C, 1, 1,} matrix.
    // Currently the mnist example is only trained in FP32 mode.
    auto input = network->addInput(INPUT_BLOB_NAME, dt, nvinfer1::Dims3{(INPUT_H * INPUT_W), 1, 1});
    assert(input != nullptr);

    for (int i = 0; i < HIDDEN_COUNT; ++i)
    {
        std::stringstream weightStr, biasStr;
        weightStr << "hiddenWeights" << i;
        biasStr << "hiddenBias" << i;
        // Transpose hidden layer weights
        transposeWeights(weightMap[weightStr.str()].data, HIDDEN_SIZE);
        auto mlpLayer = addMLPLayer(network, *input, HIDDEN_SIZE, weightMap[weightStr.str()].data, weightMap[biasStr.str()].data, MNIST_ACT, i);
        input = mlpLayer->getOutput(0);
    }
    // Transpose output layer weights
    transposeWeights(weightMap["outputWeights"].data, OUTPUT_SIZE);

    auto finalLayer = addMLPLayer(network, *input, OUTPUT_SIZE, weightMap["outputWeights"].data, weightMap["outputBias"].data, MNIST_ACT, -1);
    assert(finalLayer != nullptr);
    // Run topK to get the final result
    auto topK = network->addTopK(*finalLayer->getOutput(0), nvinfer1::TopKOperation::kMAX, 1, 0x1);
    assert(topK != nullptr);
    topK->setName("OutputTopK");
    topK->getOutput(1)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*topK->getOutput(1));
    topK->getOutput(1)->setType(nvinfer1::DataType::kINT32);

    // Build the engine
    builder->setMaxBatchSize(MAX_BATCH_SIZE);
    builder->setMaxWorkspaceSize(1 << 30);

    samplesCommon::enableDLA(builder, gUseDLACore);

    auto engine = builder->buildCudaEngine(*network);
    // we don't need the network any more
    network->destroy();

    // Once we have built the cuda engine, we can release all of our held memory.
    for (auto& mem : weightMap)
        free(const_cast<void*>(mem.second.data.values));
    return engine;
}

void APIToModel(nvinfer1::IHostMemory** modelStream)
{
    // create the builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

    // create the model to populate the network, then set the outputs and create an engine
    nvinfer1::ICudaEngine* engine = fromAPIToModel(builder);

    assert(engine != nullptr);

    // GIE-3533
    // serialize the engine, then close everything down
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}

void doInference(nvinfer1::IExecutionContext& context, uint8_t* inputPtr, uint8_t* outputPtr)
{
    float* input = reinterpret_cast<float*>(inputPtr);
    int32_t* output = reinterpret_cast<int32_t*>(outputPtr);
    const nvinfer1::ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], MAX_BATCH_SIZE * (INPUT_H * INPUT_W) * 4));
    CHECK(cudaMalloc(&buffers[outputIndex], MAX_BATCH_SIZE * 4));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, MAX_BATCH_SIZE * (INPUT_H * INPUT_W) * 4, cudaMemcpyHostToDevice, stream));
    context.enqueue(MAX_BATCH_SIZE, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], MAX_BATCH_SIZE * 4, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char* argv[])
{
    gUseDLACore = samplesCommon::parseDLA(argc, argv);
    // create a model using the API directly and serialize it to a stream.
    nvinfer1::IHostMemory* modelStream{nullptr};

    // Temporarily disable serialization path while debugging the layer.
    APIToModel(&modelStream);
    assert(modelStream != nullptr);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    if (gUseDLACore >= 0)
    {
        runtime->setDLACore(gUseDLACore);
    }
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);
    assert(engine != nullptr);
    modelStream->destroy();
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    srand(unsigned(time(nullptr)));
    bool pass{true};
    int num = rand() % 10;
    // Just for simplicity, allocations for memory use float,
    // even for fp16 data type.
    uint8_t* input = new uint8_t[MAX_BATCH_SIZE * (INPUT_H * INPUT_W) * sizeof(float)];
    uint8_t* output = new uint8_t[MAX_BATCH_SIZE * sizeof(float)];
    assert(input != nullptr);
    assert(output != nullptr);

    // read a random digit file from the data directory for use as input.
    auto fileData = new uint8_t[(INPUT_H * INPUT_W)];
    readPGMFile(std::to_string(num) + ".pgm", fileData);

    // print the ascii representation of the file that was loaded.
    std::cout << "\n\n\n---------------------------"
              << "\n\n\n"
              << std::endl;
    for (int i = 0; i < (INPUT_H * INPUT_W); i++)
        std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

    // Normalize the data the same way TensorFlow does.
    for (int i = 0; i < (INPUT_H * INPUT_W); i++)
        reinterpret_cast<float*>(input)[i] = 1.0 - float(fileData[i]) / 255.0f;

    delete[] fileData;

    doInference(*context, input, output);

    int idx{*reinterpret_cast<int*>(output)};
    std::cout << "\n\n";
    pass = (idx == num);
    if (pass)
        std::cout << "&&&& PASSED - Algorithm chose " << idx << std::endl;
    else
        std::cout << "&&&& FAILED - Algorithm chose " << idx << " but expected " << num << "." << std::endl;

    delete[] input;
    delete[] output;

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
