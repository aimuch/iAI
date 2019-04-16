#include "NvInfer.h"
#include "NvUtils.h"
#include "cuda_runtime_api.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include "common.h"

static Logger gLogger;
// To train the model that this sample uses the dataset can be found here:
// http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
//
// The ptb_w model was created retrieved from:
// https://github.com/okuchaiev/models.git
//
// The tensorflow command used to train:
// python models/tutorials/rnn/ptb/ptb_word_lm.py --data_path=data --file_prefix=ptb.char --model=charlarge --save_path=charlarge/ --seed_for_sample='consumer rep'
//
// Epochs trained: 30 
// Test perplexity: 2.697
//
// Training outputs a params.p file, which contains all of the weights in pickle format.
// This data was converted via a python script that did the following.
// Cell0 and Cell1 Linear weights matrices were concatenated as rnnweight
// Cell0 and Cell1 Linear bias vectors were concatenated as rnnbias
// Embedded is added as embed.
// softmax_w is added as rnnfcw
// softmax_b is added as rnnfcb
//
// The floating point values are converted to 32bit integer hexadecimal and written out to char-rnn.wts.

// These mappings came from training with tensorflow 0.12.1
// and emitting the word to id and id to word mappings from
// the checkpoint data after loading it.
// The only difference is that in the data set that was used,
static std::map<char, int> char_to_id{{'#', 40},
    { '$', 31}, { '\'', 28}, { '&', 35}, { '*', 49},
    { '-', 32}, { '/', 48}, { '.', 27}, { '1', 37},
    { '0', 36}, { '3', 39}, { '2', 41}, { '5', 43},
    { '4', 47}, { '7', 45}, { '6', 46}, { '9', 38},
    { '8', 42}, { '<', 22}, { '>', 23}, { '\0', 24},
    { 'N', 26}, { '\\', 44}, { ' ', 0}, { 'a', 3},
    { 'c', 13}, { 'b', 20}, { 'e', 1}, { 'd', 12},
    { 'g', 18}, { 'f', 15}, { 'i', 6}, { 'h', 9},
    { 'k', 17}, { 'j', 30}, { 'm', 14}, { 'l', 10},
    { 'o', 5}, { 'n', 4}, { 'q', 33}, { 'p', 16},
    { 's', 7}, { 'r', 8}, { 'u', 11}, { 't', 2},
    { 'w', 21}, { 'v', 25}, { 'y', 19}, { 'x', 29},
    { 'z', 34}
};

// A mapping from index to character.
static std::vector<char> id_to_char{{' ', 'e', 't', 'a',
    'n', 'o', 'i', 's', 'r', 'h', 'l', 'u', 'd', 'c',
    'm', 'f', 'p', 'k', 'g', 'y', 'b', 'w', '<', '>',
    '\0', 'v', 'N', '.', '\'', 'x', 'j', '$', '-', 'q',
    'z', '&', '0', '1', '9', '3', '#', '2', '8', '5',
    '\\', '7', '6', '4', '/', '*'}};

// Information describing the network
static const int LAYER_COUNT = 2;
static const int BATCH_SIZE = 1;
static const int HIDDEN_SIZE = 512;
static const int SEQ_SIZE = 1;
static const int DATA_SIZE = HIDDEN_SIZE;
static const int OUTPUT_SIZE = 50;

const char* INPUT_BLOB_NAME = "data";
const char* HIDDEN_IN_BLOB_NAME = "hiddenIn";
const char* CELL_IN_BLOB_NAME = "cellIn";
const char* OUTPUT_BLOB_NAME = "prob";
const char* HIDDEN_OUT_BLOB_NAME = "hiddenOut";
const char* CELL_OUT_BLOB_NAME = "cellOut";



using namespace nvinfer1;

// Our weight files are in a very simple space delimited format.
// type is the integer value of the DataType enum in NvInfer.h.
// <number of buffers>
// for each buffer: [name] [type] [size] <data x size in hex> 
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::map<std::string, Weights> weightMap;
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while(count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);
        if (wt.type == DataType::kFLOAT)
        {
            uint32_t *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];

            }
            wt.values = val;
        } else if (wt.type == DataType::kHALF)
        {
            uint16_t *val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

// We have the data files located in a specific directory. This 
// searches for that directory format from the current directory.
std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/char-rnn/", "data/char-rnn/"};
    return locateFile(input, dirs);
}

// Reshape plugin to feed RNN into FC layer correctly.
class Reshape : public IPlugin
{
public:
	Reshape(size_t size) : mSize(size) {} 
	Reshape(const void*buf, size_t size)
    {
        assert(size == sizeof(mSize));
        mSize = *static_cast<const size_t*>(buf);
    }
	int getNbOutputs() const override													{	return 1;	}
	int initialize() override															{	return 0;	}
	void terminate() override															{}
	size_t getWorkspaceSize(int) const override											{	return 0;	}
	int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        CHECK(cudaMemcpyAsync(static_cast<float*>(outputs[0]),
                   static_cast<const float*>(inputs[0]),
                   sizeof(float) * mSize * batchSize, cudaMemcpyDefault, stream));
        return 0;
    }
	size_t getSerializationSize() override
    {
        return sizeof(mSize);
    }
	void serialize(void* buffer) override
    {
        (*static_cast<size_t*>(buffer)) = mSize;

    }
	void configure(const Dims*, int, const Dims*, int, int)	override					{ }
    // The RNN outputs in {L, N, C}, but FC layer needs {C, 1, 1}, so we can convert RNN
    // output to {L*N, C, 1, 1} and TensorRT will handle the rest.
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
		return DimsNCHW(inputs[index].d[1] * inputs[index].d[0], inputs[index].d[2], 1, 1);
	}
    private:
    size_t mSize{0};
};
class PluginFactory : public nvinfer1::IPluginFactory
{
public:
	// deserialization plugin implementation
	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
        assert(!strncmp(layerName, "reshape", 7));
        if (!mPlugin) mPlugin = new Reshape(serialData, serialLength);
        return mPlugin;
    }
    void destroyPlugin()
    {
        if (mPlugin) delete mPlugin;
        mPlugin = nullptr;
    }
private:
    Reshape *mPlugin{nullptr};
}; // PluginFactory
	
// TensorFlow weight parameters for BasicLSTMCell
// are formatted as:
// Each [WR][icfo] is hiddenSize sequential elements.
// CellN  Row 0: WiT, WcT, WfT, WoT
// CellN  Row 1: WiT, WcT, WfT, WoT
// ...
// CellN RowM-1: WiT, WcT, WfT, WoT
// CellN RowM+0: RiT, RcT, RfT, RoT
// CellN RowM+1: RiT, RcT, RfT, RoT
// ...
// CellNRow2M-1: RiT, RcT, RfT, RoT
//
// TensorRT expects the format to laid out in memory:
// CellN: Wf, Wi, Wc, Wo, Rf, Ri, Rc, Ro
Weights convertRNNWeights(Weights input)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count));
    int indir[4]{ 1, 2, 0, 3 };
    int order[5]{ 0, 1, 4, 2, 3};
    int dims[5]{LAYER_COUNT, 2, 4, HIDDEN_SIZE, HIDDEN_SIZE};
    utils::reshapeWeights(input, dims, order, ptr, 5);
    utils::transposeSubBuffers(ptr, DataType::kFLOAT, LAYER_COUNT * 2, HIDDEN_SIZE * HIDDEN_SIZE, 4);
    int subMatrix = HIDDEN_SIZE * HIDDEN_SIZE;
    int layerOffset = 8 * subMatrix;
    for (int z = 0; z < LAYER_COUNT; ++z)
    {
        utils::reorderSubBuffers(ptr + z * layerOffset, indir, 4, subMatrix * sizeof(float));
        utils::reorderSubBuffers(ptr + z * layerOffset + 4 * subMatrix, indir, 4, subMatrix * sizeof(float));
    }
    return Weights{input.type, ptr, input.count};
}

// TensorFlow bias parameters for BasicLSTMCell
// are formatted as:
// CellN: Bi, Bc, Bf, Bo
//
// TensorRT expects the format to be:
// CellN: Wf, Wi, Wc, Wo, Rf, Ri, Rc, Ro
//
// Since tensorflow already combines U and W,
// we double the size and set all of U to zero.
Weights convertRNNBias(Weights input)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count*2));
    std::fill(ptr, ptr + input.count*2, 0);
    const float* iptr = static_cast<const float*>(input.values);
    int indir[4]{ 1, 2, 0, 3 };
    for (int z = 0, y = 0; z < LAYER_COUNT; ++z)
        for (int x = 0; x < 4; ++x, ++y)
            std::copy(iptr + y * HIDDEN_SIZE , iptr + (y + 1) * HIDDEN_SIZE, ptr + (z * 8 + indir[x]) * HIDDEN_SIZE);
    return Weights{input.type, ptr, input.count*2};
}

// The fully connected weights from tensorflow are transposed compared to 
// the order that tensorRT expects them to be in.
Weights transposeFCWeights(Weights input)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count));
    const float* iptr = static_cast<const float*>(input.values);
    assert(input.count == HIDDEN_SIZE * OUTPUT_SIZE);
    for (int z = 0; z < HIDDEN_SIZE; ++z)
        for (int x = 0; x < OUTPUT_SIZE; ++x)
            ptr[x * HIDDEN_SIZE + z] = iptr[z * OUTPUT_SIZE + x];
    return Weights{input.type, ptr, input.count};
}

void APIToModel(std::map<std::string, Weights> &weightMap, IHostMemory **modelStream)
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // create the model to populate the network, then set the outputs and create an engine
    INetworkDefinition* network = builder->createNetwork();

    auto data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, DimsCHW{ SEQ_SIZE, BATCH_SIZE, DATA_SIZE});
    assert(data != nullptr);

    auto hiddenIn = network->addInput(HIDDEN_IN_BLOB_NAME, DataType::kFLOAT, DimsCHW{ LAYER_COUNT, BATCH_SIZE, HIDDEN_SIZE});
    assert(hiddenIn != nullptr);

    auto cellIn = network->addInput(CELL_IN_BLOB_NAME, DataType::kFLOAT, DimsCHW{ LAYER_COUNT, BATCH_SIZE, HIDDEN_SIZE});
    assert(cellIn != nullptr);

    // Create an RNN layer w/ 2 layers and 512 hidden states
    auto tfwts = weightMap["rnnweight"];
    Weights rnnwts = convertRNNWeights(tfwts);
    auto tfbias = weightMap["rnnbias"];
    Weights rnnbias = convertRNNBias(tfbias);

    auto rnn = network->addRNN(*data, LAYER_COUNT, HIDDEN_SIZE, SEQ_SIZE,
            RNNOperation::kLSTM, RNNInputMode::kLINEAR, RNNDirection::kUNIDIRECTION,
            rnnwts, rnnbias);
    assert(rnn != nullptr);
    rnn->getOutput(0)->setName("RNN output");
    rnn->setHiddenState(*hiddenIn);
    if (rnn->getOperation() == RNNOperation::kLSTM)
        rnn->setCellState(*cellIn);
    
    Reshape reshape(SEQ_SIZE * BATCH_SIZE * HIDDEN_SIZE);
    ITensor *ptr = rnn->getOutput(0);
    auto plugin = network->addPlugin(&ptr, 1, reshape);
    plugin->setName("reshape");

    // Add a second fully connected layer with 50 outputs.
    auto tffcwts = weightMap["rnnfcw"];
    auto wts = transposeFCWeights(tffcwts);
    auto bias = weightMap["rnnfcb"];
    auto fc = network->addFullyConnected(*plugin->getOutput(0), OUTPUT_SIZE, wts, bias);
    assert(fc != nullptr);
    fc->getOutput(0)->setName("FC output");

    // Add a softmax layer to determine the probability.
    auto prob = network->addSoftMax(*fc->getOutput(0));
    assert(prob != nullptr);
    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*prob->getOutput(0));
    rnn->getOutput(1)->setName(HIDDEN_OUT_BLOB_NAME);
    network->markOutput(*rnn->getOutput(1));
    if (rnn->getOperation() == RNNOperation::kLSTM)
    {
        rnn->getOutput(2)->setName(CELL_OUT_BLOB_NAME);
        network->markOutput(*rnn->getOutput(2));
    }

    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 25);

    // Store the transformed weights in the weight map so the memory can be properly released later.
    weightMap["rnnweight2"] = rnnwts;
    weightMap["rnnbias2"] = rnnbias;
    weightMap["rnnfcw2"] = wts;

    auto engine = builder->buildCudaEngine(*network);
    assert(engine != nullptr);
    // we don't need the network any more
    network->destroy();

    // serialize the engine, then close everything down
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}

void stepOnce(float **data, void **buffers, int *sizes, int *indices,
        int numBindings, cudaStream_t &stream, IExecutionContext &context)
{
    for (int z = 0, w = numBindings/2; z < w; ++z)
        CHECK(cudaMemcpyAsync(buffers[indices[z]], data[z], sizes[z] * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Execute asynchronously
    context.enqueue(1, buffers, stream, nullptr);

    // DMA the input from the GPU
    for (int z = numBindings/2, w = numBindings; z < w; ++z)
        CHECK(cudaMemcpyAsync(data[z], buffers[indices[z]], sizes[z] * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // Copy Ct/Ht to the Ct-1/Ht-1 slots.
    CHECK(cudaMemcpyAsync(data[1], buffers[indices[4]], sizes[1] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(data[2], buffers[indices[5]], sizes[2] * sizeof(float), cudaMemcpyDeviceToHost, stream));
}

bool doInference(IExecutionContext& context, std::string input, std::string expected, std::map<std::string, Weights> &weightMap)
{
    const ICudaEngine& engine = context.getEngine();
    // We have 6 outputs for LSTM, this needs to be changed to 4 for any other RNN type
    static const int numBindings = 6;
    assert(engine.getNbBindings() == numBindings);
    void* buffers[numBindings];
    float* data[numBindings];
    std::fill(buffers, buffers + numBindings, nullptr);
    std::fill(data, data + numBindings, nullptr);
    const char *names[numBindings] = {INPUT_BLOB_NAME,
        HIDDEN_IN_BLOB_NAME,
        CELL_IN_BLOB_NAME,
        OUTPUT_BLOB_NAME,
        HIDDEN_OUT_BLOB_NAME,
        CELL_OUT_BLOB_NAME
    };
    int indices[numBindings];
    std::fill(indices, indices + numBindings, -1);
    int sizes[numBindings] = { SEQ_SIZE * BATCH_SIZE * DATA_SIZE,
        LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE,
        LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE,
        OUTPUT_SIZE,
        LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE,
        LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE
    };

    for (int x = 0; x < numBindings; ++x)
    {
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // note that indices are guaranteed to be less than IEngine::getNbBindings()
        indices[x] = engine.getBindingIndex(names[x]);
        if (indices[x] == -1) continue;
        // create GPU buffers and a stream
        assert(indices[x] < numBindings);
        CHECK(cudaMalloc(&buffers[indices[x]], sizes[x] * sizeof(float)));
        data[x] = new float[sizes[x]];
    }
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // Initialize input/hidden/cell state to zero
    for (int x = 0; x < numBindings; ++x) std::fill(data[x], data[x] + sizes[x], 0.0f);

    auto embed = weightMap["embed"];
    std::string genstr;
    assert(BATCH_SIZE == 1 && "This code assumes batch size is equal to 1.");
    // Seed the RNN with the input.
    for (auto &a : input)
    {
        std::copy(reinterpret_cast<const float*>(embed.values) + char_to_id[a]*DATA_SIZE,
                reinterpret_cast<const float*>(embed.values) + char_to_id[a]*DATA_SIZE + DATA_SIZE,
                data[0]);
        stepOnce(data, buffers, sizes, indices, 6, stream, context);
        cudaStreamSynchronize(stream);
        genstr.push_back(a);
    }
    // Now that we have gone through the initial sequence, lets make sure that we get the sequence out that
    // we are expecting.
    for (size_t x = 0, y = expected.size(); x < y; ++x)
    {
        std::copy(reinterpret_cast<const float*>(embed.values) + char_to_id[*genstr.rbegin()]*DATA_SIZE,
                reinterpret_cast<const float*>(embed.values) + char_to_id[*genstr.rbegin()]*DATA_SIZE + DATA_SIZE,
                data[0]);

        stepOnce(data, buffers, sizes, indices, 6, stream, context);
        cudaStreamSynchronize(stream);

		float* probabilities = reinterpret_cast<float*>(data[indices[3]]);
		ptrdiff_t idx = std::max_element(probabilities, probabilities + sizes[3]) - probabilities;
        genstr.push_back(id_to_char[idx]);
    }
    printf("Received: %s\n", genstr.c_str() + input.size());

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    for (int x = 0; x < numBindings; ++x)
    {
        CHECK(cudaFree(buffers[indices[x]]));
        if (data[x]) delete [] data[x];
    }
    return genstr == (input + expected);
}

int main(int argc, char** argv)
{
    // create a model using the API directly and serialize it to a stream
    IHostMemory *modelStream{nullptr};

    std::map<std::string, Weights> weightMap = loadWeights(locateFile("char-rnn.wts"));
    APIToModel(weightMap, &modelStream);

    srand(unsigned(time(nullptr)));
    const char* strings[10]{ "customer serv",
        "business plans",
        "help",
        "slightly under",
        "market",
        "holiday cards",
        "bring it",
        "what time",
        "the owner thinks",
        "money can be use"
    };
    const char* outs[10]{ "es and the",
        " to be a",
        "en and",
        "iting the company",
        "ing and",
        " the company",
        " company said it will",
        "d and the company",
        "ist with the",
        "d to be a"
    };
    // Select a random seed string.
    int num = rand() % 10;
    PluginFactory pluginFactory;

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), &pluginFactory);
    if (modelStream) modelStream->destroy();

    IExecutionContext *context = engine->createExecutionContext();

    bool pass {false};
    std::cout << "\n---------------------------" << "\n";
    std::cout << "RNN Warmup: " << strings[num] << std::endl;
    std::cout << "Expect: " << outs[num] << std::endl;
    pass = doInference(*context, strings[num], outs[num], weightMap);
    if (!pass) std::cout << "Failure!" << std::endl;
    std::cout << "---------------------------" << "\n";

    for (auto &mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    pluginFactory.destroyPlugin();
    return !pass ? EXIT_FAILURE : EXIT_SUCCESS;
}
