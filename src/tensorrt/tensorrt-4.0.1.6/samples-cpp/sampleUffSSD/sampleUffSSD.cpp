#include <cassert>
#include <chrono>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <time.h>
#include <unordered_map>
#include <vector>

#include "BatchStreamPPM.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include "common.h"

using namespace nvinfer1;
using namespace nvuffparser;
using namespace plugin;

static Logger gLogger;
static samples_common::Args args;

#define RETURN_AND_LOG(ret, severity, message)                                 \
    do                                                                         \
    {                                                                          \
        std::string error_message = "sample_uff_ssd: " + std::string(message); \
        gLogger.log(ILogger::Severity::k##severity, error_message.c_str());    \
        return (ret);                                                          \
    } while (0)

static constexpr int OUTPUT_CLS_SIZE = 91;
static constexpr int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

const char* OUTPUT_BLOB_NAME0 = "NMS";

//INT8 Calibration, currently set to calibrate over 500 images
static constexpr int CAL_BATCH_SIZE = 50;
static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;

// Concat layers
// mbox_priorbox, mbox_loc, mbox_conf
const int concatAxis[2] = {1, 1};
const bool ignoreBatch[2] = {false, false};

DetectionOutputParameters detectionOutputParam{true, false, 0, OUTPUT_CLS_SIZE, 200, 100, 0.5, 0.6, CodeTypeSSD::TF_CENTER, {0, 2, 1}, true, true};

// Visualization
const float visualizeThreshold = 0.5;

void printOutput(int64_t eltCount, DataType dtype, void* buffer)
{
    std::cout << eltCount << " eltCount" << std::endl;
    assert(samples_common::getElementSize(dtype) == sizeof(float));
    std::cout << "--- OUTPUT ---" << std::endl;

    size_t memSize = eltCount * samples_common::getElementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpyAsync(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = std::distance(outputs, std::max_element(outputs, outputs + eltCount));

    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        std::cout << eltIdx << " => " << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx)
            std::cout << "***";
        std::cout << "\n";
    }

    std::cout << std::endl;
    delete[] outputs;
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/ssd/",
                                  "data/ssd/VOC2007/",
                                  "data/ssd/VOC2007/PPMImages/",
                                  "data/samples/ssd/",
                                  "data/samples/ssd/VOC2007/",
                                  "data/samples/ssd/VOC2007/PPMImages/"};
    return locateFile(input, dirs);
}

void populateTFInputData(float* data)
{

    auto fileName = locateFile("inp_bus.txt");
    std::ifstream labelFile(fileName);
    string line;
    int id = 0;
    while (getline(labelFile, line))
    {
        istringstream iss(line);
        float num;
        iss >> num;
        data[id++] = num;
    }

    return;
}

void populateClassLabels(std::string (&CLASSES)[OUTPUT_CLS_SIZE])
{

    auto fileName = locateFile("ssd_coco_labels.txt");
    std::ifstream labelFile(fileName);
    string line;
    int id = 0;
    while (getline(labelFile, line))
    {
        CLASSES[id++] = line;
    }

    return;
}

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = samples_common::volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser, nvuffparser::IPluginFactory* pluginFactory,
                                      IInt8Calibrator* calibrator, IHostMemory*& trtModelStream)
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Parse the UFF model to populate the network, then set the outputs.
    INetworkDefinition* network = builder->createNetwork();
    parser->setPluginFactory(pluginFactory);

    std::cout << "Begin parsing model..." << std::endl;
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");

    std::cout << "End parsing model..." << std::endl;

    // Build the engine.
    builder->setMaxBatchSize(maxBatchSize);
    // The _GB literal operator is defined in common/common.h
    builder->setMaxWorkspaceSize(1_GB); // We need about 1GB of scratch space for the plugin layer for batch size 5.
    builder->setHalf2Mode(false);
    if (args.runInInt8)
    {
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator);
    }

    std::cout << "Begin building engine..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
    std::cout << "End building engine..." << std::endl;

    // We don't need the network any more, and we can destroy the parser.
    network->destroy();
    parser->destroy();

    // Serialize the engine, then close everything down.
    trtModelStream = engine->serialize();

    builder->destroy();
    shutdownProtobufLibrary();
    return engine;
}

void doInference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    int nbBindings = engine.getNbBindings();

    std::vector<void*> buffers(nbBindings);
    std::vector<std::pair<int64_t, DataType>> buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    for (int i = 0; i < nbBindings; ++i)
    {
        auto bufferSizesOutput = buffersSizes[i];
        buffers[i] = samples_common::safeCudaMalloc(bufferSizesOutput.first * samples_common::getElementSize(bufferSizesOutput.second));
    }

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
        outputIndex1 = outputIndex0 + 1; //engine.getBindingIndex(OUTPUT_BLOB_NAME1);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

    auto t_start = std::chrono::high_resolution_clock::now();
    context.execute(batchSize, &buffers[0]);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    std::cout << "Time taken for inference is " << total << " ms." << std::endl;

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
    {
        if (engine.bindingIsInput(bindingIdx))
            continue;
#ifdef SSD_INT8_DEBUG
        auto bufferSizesOutput = buffersSizes[bindingIdx];
        printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                    buffers[bindingIdx]);
#endif
    }

    CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * detectionOutputParam.keepTopK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(keepCount, buffers[outputIndex1], batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

class FlattenConcat : public IPlugin
{
public:
    FlattenConcat(int concatAxis, bool ignoreBatch)
        : mIgnoreBatch(ignoreBatch)
        , mConcatAxisID(concatAxis)
    {
        assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
    }

    FlattenConcat(const void* data, size_t length)
    {
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        mIgnoreBatch = read<bool>(d);
        mConcatAxisID = read<int>(d);
        assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
        mOutputConcatAxis = read<int>(d);
        mNumInputs = read<int>(d);
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        CHECK(cudaMallocHost((void**) &mCopySize, mNumInputs * sizeof(int)));

        std::for_each(mInputConcatAxis, mInputConcatAxis + mNumInputs, [&](int& inp) { inp = read<int>(d); });

        mCHW = read<nvinfer1::DimsCHW>(d);

        std::for_each(mCopySize, mCopySize + mNumInputs, [&](size_t& inp) { inp = read<size_t>(d); });

        assert(d == a + length);
    }
    ~FlattenConcat()
    {
        CHECK(cudaFreeHost(mInputConcatAxis));
        CHECK(cudaFreeHost(mCopySize));
    }
    int getNbOutputs() const override { return 1; }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims >= 1);
        assert(index == 0);
        mNumInputs = nbInputDims;
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        mOutputConcatAxis = 0;
#ifdef SSD_INT8_DEBUG
        std::cout << " Concat nbInputs " << nbInputDims << "\n";
        std::cout << " Concat axis " << mConcatAxisID << "\n";
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 3; ++j)
                std::cout << " Concat InputDims[" << i << "]"
                          << "d[" << j << " is " << inputs[i].d[j] << "\n";
#endif
        for (int i = 0; i < nbInputDims; ++i)
        {
            int flattenInput = 0;
            assert(inputs[i].nbDims == 3);
            if (mConcatAxisID != 1) assert(inputs[i].d[0] == inputs[0].d[0]);
            if (mConcatAxisID != 2) assert(inputs[i].d[1] == inputs[0].d[1]);
            if (mConcatAxisID != 3) assert(inputs[i].d[2] == inputs[0].d[2]);
            flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
            mInputConcatAxis[i] = flattenInput;
            mOutputConcatAxis += mInputConcatAxis[i];
        }

        return DimsCHW(mConcatAxisID == 1 ? mOutputConcatAxis : 1,
                       mConcatAxisID == 2 ? mOutputConcatAxis : 1,
                       mConcatAxisID == 3 ? mOutputConcatAxis : 1);
    }

    int initialize() override
    {
        CHECK(cublasCreate(&mCublas));
        return 0;
    }

    void terminate() override
    {
        CHECK(cublasDestroy(mCublas));
    }

    size_t getWorkspaceSize(int) const override { return 0; }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
    {
        int numConcats = 1;
        assert(mConcatAxisID != 0);
        numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1, std::multiplies<int>());

        if (!mIgnoreBatch) numConcats *= batchSize;

        float* output = reinterpret_cast<float*>(outputs[0]);
        int offset = 0;
        for (int i = 0; i < mNumInputs; ++i)
        {
            const float* input = reinterpret_cast<const float*>(inputs[i]);
            float* inputTemp;
            CHECK(cudaMalloc(&inputTemp, mCopySize[i] * batchSize));

            CHECK(cudaMemcpyAsync(inputTemp, input, mCopySize[i] * batchSize, cudaMemcpyDeviceToDevice, stream));

            for (int n = 0; n < numConcats; ++n)
            {
                CHECK(cublasScopy(mCublas, mInputConcatAxis[i],
                                  inputTemp + n * mInputConcatAxis[i], 1,
                                  output + (n * mOutputConcatAxis + offset), 1));
            }
            CHECK(cudaFree(inputTemp));
            offset += mInputConcatAxis[i];
        }

        return 0;
    }

    size_t getSerializationSize() override
    {
        return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims) + (sizeof(mCopySize) * mNumInputs);
    }

    void serialize(void* buffer) override
    {
        char *d = reinterpret_cast<char *>(buffer), *a = d;
        write(d, mIgnoreBatch);
        write(d, mConcatAxisID);
        write(d, mOutputConcatAxis);
        write(d, mNumInputs);
        for (int i = 0; i < mNumInputs; ++i)
        {
            write(d, mInputConcatAxis[i]);
        }
        write(d, mCHW);
        for (int i = 0; i < mNumInputs; ++i)
        {
            write(d, mCopySize[i]);
        }
        assert(d == a + getSerializationSize());
    }

    void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
        assert(nbOutputs == 1);
        mCHW = inputs[0];
        assert(inputs[0].nbDims == 3);
        CHECK(cudaMallocHost((void**) &mCopySize, nbInputs * sizeof(int)));
        for (int i = 0; i < nbInputs; ++i)
        {
            mCopySize[i] = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2] * sizeof(float);
        }
    }

private:
    template <typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    size_t* mCopySize;
    bool mIgnoreBatch{false};
    int mConcatAxisID, mOutputConcatAxis, mNumInputs;
    int* mInputConcatAxis;
    nvinfer1::Dims mCHW;
    cublasHandle_t mCublas;
};

// Integration for serialization.
class PluginFactory : public nvinfer1::IPluginFactory, public nvuffparser::IPluginFactory
{
public:
    std::unordered_map<std::string, int> concatIDs = {
        std::make_pair("_concat_box_loc", 0),
        std::make_pair("_concat_box_conf", 1)};

        virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights, const nvuffparser::FieldCollection fc) override
        {
            assert(isPlugin(layerName));

            const nvuffparser::FieldMap* fields = fc.fields;
            int nbFields = fc.nbFields;

            if(!strcmp(layerName, "_PriorBox"))
            {
                assert(mPluginPriorBox == nullptr);
                assert(nbWeights == 0 && weights == nullptr);

                float minScale = 0.2, maxScale = 0.95;
                int numLayers;
                std::vector<float> aspectRatios;
                std::vector<int> fMapShapes;
                std::vector<float> layerVariances;

                for(int i = 0; i < nbFields; i++)
                {
                    const char* attr_name = fields[i].name;
                    if (strcmp(attr_name, "numLayers") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        numLayers = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "minScale") == 0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        minScale = (float)(*(static_cast<const double*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "maxScale") == 0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        maxScale = (float)(*(static_cast<const double*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "aspectRatios")==0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        int size = fields[i].length;
                        aspectRatios.reserve(size);
                        const double *aR = static_cast<const double*>(fields[i].data);
                        for(int j=0; j < size; j++)
                        {
                            aspectRatios.push_back(*aR);
                            aR++;
                        }
                    }
                    else if(strcmp(attr_name, "featureMapShapes")==0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        int size = fields[i].length;
                        fMapShapes.reserve(size);
                        const int *fMap = static_cast<const int*>(fields[i].data);
                        for(int j=0; j<size; j++){
                            fMapShapes.push_back(*fMap);
                            fMap++;
                        }
                    }
                    else if(strcmp(attr_name, "layerVariances")==0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        int size = fields[i].length;
                        layerVariances.reserve(size);
                        const double *lVar = static_cast<const double*>(fields[i].data);
                        for(int j=0; j<size; j++){
                            layerVariances.push_back(*lVar);
                            lVar++;
                        }
                    }
                }
                // Num layers should match the number of feature maps from which boxes are predicted.
                assert(numLayers > 0);
                assert((int)fMapShapes.size() == numLayers);
                assert(aspectRatios.size() > 0);
                assert(layerVariances.size() == 4);

                // Reducing the number of boxes predicted by the first layer.
                // This is in accordance with the standard implementation.
                vector<float> firstLayerAspectRatios;

                int numFirstLayerARs = 3;
                for(int i = 0; i < numFirstLayerARs; ++i){
                    firstLayerAspectRatios.push_back(aspectRatios[i]);
                }
                // A comprehensive list of box parameters that are required by anchor generator
                GridAnchorParameters boxParams[numLayers];
                for(int i = 0; i < numLayers ; i++)
                {
                    if(i == 0)
                        boxParams[i] = {minScale, maxScale, firstLayerAspectRatios.data(), (int)firstLayerAspectRatios.size(), fMapShapes[i], fMapShapes[i], {layerVariances[0], layerVariances[1], layerVariances[2], layerVariances[3]}};
                    else
                        boxParams[i] = {minScale, maxScale, aspectRatios.data(), (int)aspectRatios.size(), fMapShapes[i], fMapShapes[i], {layerVariances[0], layerVariances[1], layerVariances[2], layerVariances[3]}};
                }

                mPluginPriorBox = std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createSSDAnchorGeneratorPlugin(boxParams, numLayers), nvPluginDeleter);
                return mPluginPriorBox.get();
            }
            else if(concatIDs.find(std::string(layerName)) != concatIDs.end())
            {
                const int i = concatIDs[layerName];
                assert(mPluginFlattenConcat[i] == nullptr);
                assert(nbWeights == 0 && weights == nullptr);
                mPluginFlattenConcat[i] = std::unique_ptr<FlattenConcat>(new FlattenConcat(concatAxis[i], ignoreBatch[i]));
                return mPluginFlattenConcat[i].get();
            }
            else if(!strcmp(layerName, "_concat_priorbox"))
            {
                assert(mPluginConcat == nullptr);
                assert(nbWeights == 0 && weights == nullptr);
                mPluginConcat = std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createConcatPlugin(2, true), nvPluginDeleter);
                return mPluginConcat.get();
            }
            else if(!strcmp(layerName, "_NMS"))
            {

                assert(mPluginDetectionOutput == nullptr);
                assert(nbWeights == 0 && weights == nullptr);

                 // Fill the custom attribute values to the built-in plugin according to the types
                for(int i = 0; i < nbFields; ++i)
                {
                    const char* attr_name = fields[i].name;
                    if (strcmp(attr_name, "iouThreshold") == 0)
                    {
                        detectionOutputParam.nmsThreshold =(float)(*(static_cast<const double*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "numClasses") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        detectionOutputParam.numClasses = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "maxDetectionsPerClass") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        detectionOutputParam.topK = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "scoreConverter") == 0)
                    {
                        std::string scoreConverter(static_cast<const char*>(fields[i].data), fields[i].length);
                        if(scoreConverter=="SIGMOID")
                            detectionOutputParam.confSigmoid = true;
                    }
                    else if(strcmp(attr_name, "maxTotalDetections") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        detectionOutputParam.keepTopK = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "scoreThreshold") == 0)
                    {
                        detectionOutputParam.confidenceThreshold = (float)(*(static_cast<const double*>(fields[i].data)));
                    }
                }
                mPluginDetectionOutput = std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createSSDDetectionOutputPlugin(detectionOutputParam), nvPluginDeleter);
                return mPluginDetectionOutput.get();
            }
            else
            {
              assert(0);
              return nullptr;
            }
        }

    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
    {
        assert(isPlugin(layerName));

        if (!strcmp(layerName, "_PriorBox"))
        {
            assert(mPluginPriorBox == nullptr);
            mPluginPriorBox = std::unique_ptr<INvPlugin, void (*)(INvPlugin*)>(createSSDAnchorGeneratorPlugin(serialData, serialLength), nvPluginDeleter);
            return mPluginPriorBox.get();
        }
        else if (concatIDs.find(std::string(layerName)) != concatIDs.end())
        {
            const int i = concatIDs[layerName];
            assert(mPluginFlattenConcat[i] == nullptr);
            mPluginFlattenConcat[i] = std::unique_ptr<FlattenConcat>(new FlattenConcat(serialData, serialLength));
            return mPluginFlattenConcat[i].get();
        }
        else if (!strcmp(layerName, "_concat_priorbox"))
        {
            assert(mPluginConcat == nullptr);
            mPluginConcat = std::unique_ptr<INvPlugin, void (*)(INvPlugin*)>(createConcatPlugin(serialData, serialLength), nvPluginDeleter);
            return mPluginConcat.get();
        }
        else if (!strcmp(layerName, "_NMS"))
        {
            assert(mPluginDetectionOutput == nullptr);
            mPluginDetectionOutput = std::unique_ptr<INvPlugin, void (*)(INvPlugin*)>(createSSDDetectionOutputPlugin(serialData, serialLength), nvPluginDeleter);
            return mPluginDetectionOutput.get();
        }
        else
        {
            assert(0);
            return nullptr;
        }
    }

    bool isPlugin(const char* name) override
    {
        return !strcmp(name, "_PriorBox")
            || concatIDs.find(std::string(name)) != concatIDs.end()
            || !strcmp(name, "_concat_priorbox")
            || !strcmp(name, "_NMS")
            || !strcmp(name, "mbox_conf_reshape");
    }

    // The application has to destroy the plugin when it knows it's safe to do so.
    void destroyPlugin()
    {
        for (unsigned i = 0; i < concatIDs.size(); ++i)
        {
            mPluginFlattenConcat[i].reset();
        }
        mPluginConcat.reset();
        mPluginPriorBox.reset();
        mPluginDetectionOutput.reset();
    }

    void (*nvPluginDeleter)(INvPlugin*){[](INvPlugin* ptr) { ptr->destroy(); }};
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> mPluginPriorBox{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> mPluginDetectionOutput{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> mPluginConcat{nullptr, nvPluginDeleter};
    std::unique_ptr<FlattenConcat> mPluginFlattenConcat[2]{nullptr, nullptr};
};

int main(int argc, char* argv[])
{
    // Parse command-line arguments.
    samples_common::parseArgs(args, argc, argv);

    auto fileName = locateFile("sample_ssd.uff");
    std::cout << fileName << std::endl;

    const int N = 2;
    auto parser = createUffParser();

    BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);

    parser->registerInput("Input", DimsCHW(3, 300, 300), UffInputOrder::kNCHW);
    parser->registerOutput("MarkOutput_0");

    IHostMemory* trtModelStream{nullptr};

    Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH, "CalibrationTableSSD");

    PluginFactory pluginFactorySerialize;
    ICudaEngine* tmpEngine = loadModelAndCreateEngine(fileName.c_str(), N, parser, &pluginFactorySerialize, &calibrator, trtModelStream);
    assert(tmpEngine != nullptr);
    assert(trtModelStream != nullptr);
    tmpEngine->destroy();
    pluginFactorySerialize.destroyPlugin();

    // Read a random sample image.
    srand(unsigned(time(nullptr)));
    // Available images.
    std::vector<std::string> imageList = {"dog.ppm", "bus.ppm"};
    std::vector<samples_common::PPM<INPUT_C, INPUT_H, INPUT_W>> ppms(N);

    assert(ppms.size() <= imageList.size());
    std::cout << " Num batches  " << N << std::endl;
    for (int i = 0; i < N; ++i)
    {
        readPPMFile(imageList[i], ppms[i]);
    }

    vector<float> data(N * INPUT_C * INPUT_H * INPUT_W);

    for (int i = 0, volImg = INPUT_C * INPUT_H * INPUT_W; i < N; ++i)
    {
        for (int c = 0; c < INPUT_C; ++c)
        {
            for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j) {
                data[i * volImg + c * volChl + j] = (2.0 / 255.0) * float(ppms[i].buffer[j * INPUT_C + c]) - 1.0;
            }
        }
    }
    std::cout << " Data Size  " << data.size() << std::endl;

    // Deserialize the engine.
    std::cout << "*** deserializing" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    PluginFactory pluginFactory;
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), &pluginFactory);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Host memory for outputs.
    vector<float> detectionOut(N * detectionOutputParam.keepTopK * 7);
    vector<int> keepCount(N);

    // Run inference.
    doInference(*context, &data[0], &detectionOut[0], &keepCount[0], N);
    cout << " KeepCount " << keepCount[0] << "\n";

    std::string CLASSES[OUTPUT_CLS_SIZE];

    populateClassLabels(CLASSES);

    for (int p = 0; p < N; ++p)
    {
        for (int i = 0; i < keepCount[p]; ++i)
        {
            float* det = &detectionOut[0] + (p * detectionOutputParam.keepTopK + i) * 7;
            if (det[2] < visualizeThreshold) continue;

            // Output format for each detection is stored in the below order
            // [image_id, label, confidence, xmin, ymin, xmax, ymax]
            assert((int) det[1] < OUTPUT_CLS_SIZE);
            std::string storeName = CLASSES[(int) det[1]] + "-" + std::to_string(det[2]) + ".ppm";

            printf("Detected %s in the image %d (%s) with confidence %f%% and coordinates (%f,%f),(%f,%f).\nResult stored in %s.\n", CLASSES[(int) det[1]].c_str(), int(det[0]), ppms[p].fileName.c_str(), det[2] * 100.f, det[3] * INPUT_W, det[4] * INPUT_H, det[5] * INPUT_W, det[6] * INPUT_H, storeName.c_str());

            samples_common::writePPMFileWithBBox(storeName, ppms[p], {det[3] * INPUT_W, det[4] * INPUT_H, det[5] * INPUT_W, det[6] * INPUT_H});
        }
    }

    // Destroy the engine.
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Destroy plugins created by factory
    pluginFactory.destroyPlugin();

    return EXIT_SUCCESS;
}
