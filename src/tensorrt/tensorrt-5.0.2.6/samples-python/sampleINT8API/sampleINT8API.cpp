//! sampleINT8API.cpp
//! This file contains implementation showcasing usage of INT8 calibration and precision APIs.
//! It creates classification networks such as mobilenet, vgg19, resnet-50 from onnx model file.
//! This sample showcae setting per tensor dynamic range overriding calibrator generated scales if it exists.
//! This sample showcase how to set computation precision of layer. It involves forcing output tensor type of the layer to particular precision.
//! It can be run with the following command line:
//! Command: ./sample_int8_api [-h or --help] [-m modelfile] [-s per_tensor_dynamic_range_file] [-i image_file] [-r reference_file] [-d path/to/data/dir] [--verbose] [-useDLA <id>]

#include "common.h"
#include "buffers.h"
#include "argsParser.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <unordered_map>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

static Logger gLogger(Logger::Severity::kERROR);
static const int kINPUT_C = 3;
static const int kINPUT_H = 224;
static const int kINPUT_W = 224;

// Preprocessing values are available here: https://github.com/onnx/models/tree/master/models/image_classification/resnet
static const float kMean[3] = {0.485f, 0.456f, 0.406f};
static const float kStdDev[3] = {0.229f, 0.224f, 0.225f};
static const float kScale = 255.0f;

//!
//! \brief The SampleINT8APIParams structure groups the additional parameters required by
//!         the INT8 API sample
//!
struct SampleINT8APIParams
{
    bool verbose{false};
    int dlaCore{-1};
    int batchSize;
    std::string modelFileName;
    vector<std::string> dataDirs;
    std::string perTensorDynamicRange;
    std::string imageFileName;
    std::string referenceFileName;
};

//!
//! \brief The SampleINT8APIArgs structures groups the additional arguments required by
//!         the INT8 API sample
//!
struct SampleINT8APIArgs : public samplesCommon::Args
{
    bool verbose{false};
    std::string modelFileName{"resnet50.onnx"};
    std::string imageFileName{"airliner.ppm"};
    std::string referenceFileName{"reference_labels.txt"};
    std::string perTensorDynamicRange{"resnet50_per_tensor_dynamic_range.txt"};
};

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_int8_api [-h or --help] [-m modelfile] [-s per_tensor_dynamic_range_file] [-i image_file] [-r reference_file] [-d path/to/data/dir] [--verbose] [--useDLACore=<int>]\n";
    std::cout << "-h or --help  Display This help information\n";
    std::cout << "-m  Image classification modelfile. Default : resnet50.onnx";
    std::cout << "-i  Image to infer. Defaults to data/int8/api/airlines.ppm" << std::endl;
    std::cout << "-r  Reference labels file. Defaults to data/int8/reference_labels.txt" << std::endl;
    std::cout << "-s  Specify custom per tensor dynamic range for the network. Defaults to data/int8/resnet50_per_tensor_dynamic_range.txt" << std::endl;
    std::cout << "-d  Specify data directory to search for above files. Defaults to data/samples/int8_api/" << std::endl;
    std::cout << "--verbose Outputs per tensor dynamic range and layer precision info for the network" << std::endl;
    std::cout << "--useDLACore=N   Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
}

//!
//! \brief This function parses arguments specific to sampleINT8API
//!
void parsesampleINT8APIArgs(SampleINT8APIArgs& args, int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string argStr(argv[i]);
        if (argStr == "-m")
        {
            i++;
            args.modelFileName = argv[i];
        }
        else if (argStr == "-i")
        {
            i++;
            args.imageFileName = argv[i];
        }
        else if (argStr == "-r")
        {
            i++;
            args.referenceFileName = argv[i];
        }
        else if (argStr == "-s")
        {
            i++;
            args.perTensorDynamicRange = argv[i];
        }
        else if (argStr == "--verbose")
        {
            args.verbose = true;
        }
        else if (argStr == "--help" || argStr == "-h")
        {
            args.help = true;
        }
        else if (argStr == "--useDLACore")
        {
            i++;
            args.useDLACore = std::stoi(argv[i]);
        }
        else if (argStr == "-d")
        {
            i++;
            std::string dirPath = argv[i];
            if (dirPath.back() != '/')
                dirPath.push_back('/');
            args.dataDirs.push_back(dirPath);
        }
        else
        {
            std::cout << "Invalid Argument: " << argStr << std::endl;
        }
    }
}

//!
//! \brief This function initializes members of the params struct using the command line args
//!
SampleINT8APIParams initializeSampleParams(SampleINT8APIArgs args)
{
    SampleINT8APIParams params;
    if (args.dataDirs.size() != 0) //!< Use the data directory provided by the user
        params.dataDirs = args.dataDirs;
    else //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/samples/int8_api/");
    }

    params.batchSize = 1;
    params.verbose = args.verbose;
    params.modelFileName = args.modelFileName;
    params.imageFileName = args.imageFileName;
    params.referenceFileName = args.referenceFileName;
    params.perTensorDynamicRange = args.perTensorDynamicRange;
    params.dlaCore = args.useDLACore;

    return params;
}

//!
//! \brief The sampleINT8API class implements INT8 inference on classification networks.
//!
//! \details INT8 API usage for setting custom int8 range for each input layer. API showcase how
//!           to perform INT8 inference without calibration table
//!
class sampleINT8API
{
private:
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    sampleINT8API(const SampleINT8APIParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief This function runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief This function can be used to clean up any state created in the sample class
    //!
    bool teardown();

    SampleINT8APIParams mParams; //!< Stores Sample Parameter

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr; //!< The TensorRT engine used to run the network

    std::map<std::string, std::string> mInOut; //!< Input and output mapping of the network

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network

    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network

    std::unordered_map<std::string, float> mPerTensorDynamicRangeMap; //!< Mapping from tensor name to max absolute dynamic range values

    void getInputOutputNames(); //!< Populates input and output mapping of the network

    //!
    //! \brief Reads the ppm input image, preprocesses, and stores the result in a managed buffer
    //!
    bool prepareInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Verifies that the output is correct and prints it
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers) const;

    //!
    //! \brief Populate per tensor dynamic range values
    //!
    void readPerTensorDynamicRangeValues();

    //!
    //! \brief  Sets custom dynamic range for network tensors
    //!
    void setDynamicRange(SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief  Sets computation precision for network layers
    //!
    void setLayerPrecision(SampleUniquePtr<nvinfer1::INetworkDefinition>& network);
};

//!
//! \brief  Populates input and output mapping of the network
//!
void sampleINT8API::getInputOutputNames()
{
    int nbindings = mEngine.get()->getNbBindings();
    assert(nbindings == 2);
    for (int b = 0; b < nbindings; ++b)
    {
        nvinfer1::Dims dims = mEngine.get()->getBindingDimensions(b);
        assert(dims.nbDims == 3);
        if (mEngine.get()->bindingIsInput(b))
        {
            if (mParams.verbose)
                std::cout << "Found input: "
                          << mEngine.get()->getBindingName(b)
                          << " shape=" << dims.d[0] << "," << dims.d[1] << "," << dims.d[2]
                          << " dtype=" << (int) mEngine.get()->getBindingDataType(b)
                          << std::endl;
            mInOut["input"] = mEngine.get()->getBindingName(b);
        }
        else
        {
            if (mParams.verbose)
                std::cout << "Found output: "
                          << mEngine.get()->getBindingName(b)
                          << " shape=" << dims.d[0] << "," << dims.d[1] << "," << dims.d[2]
                          << " dtype=" << (int) mEngine.get()->getBindingDataType(b)
                          << std::endl;
            mInOut["output"] = mEngine.get()->getBindingName(b);
        }
    }
}

//!
//! \brief Populate per tensor dyanamic range values
//!
void sampleINT8API::readPerTensorDynamicRangeValues()
{
    std::ifstream iDynamicRangeStream(locateFile(mParams.perTensorDynamicRange, mParams.dataDirs));
    std::string line;
    char delim = ':';
    while (std::getline(iDynamicRangeStream, line))
    {
        std::istringstream iline(line);
        std::string token;
        std::getline(iline, token, delim);
        std::string tensorName = token;
        std::getline(iline, token, delim);
        float dynamicRange = std::stof(token);
        mPerTensorDynamicRangeMap[tensorName] = dynamicRange;
    }
}

//!
//! \brief  Sets computation precision for network layers
//!
void sampleINT8API::setLayerPrecision(SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    std::cout << "[INFO] Setting Per Layer Computation Precision" << std::endl;
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        auto layer = network->getLayer(i);
        if (mParams.verbose)
        {
            std::string layerName = layer->getName();
            std::cout << "Layer: " << layerName << ". Precision: INT8 " << std::endl;
        }
        // set computation precision of the layer
        layer->setPrecision(nvinfer1::DataType::kINT8);

        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            std::string tensorName = layer->getOutput(j)->getName();
            if (mParams.verbose)
            {
                std::string tensorName = layer->getOutput(j)->getName();
                std::cout << "Tensor: " << tensorName << ". OutputType: INT8 " << std::endl;
            }
            // set output type of the tensor
            layer->setOutputType(j, nvinfer1::DataType::kINT8);
        }
    }
    std::cout << std::endl;
}

//!
//! \brief  Sets custom dynamic range for network tensors
//!
void sampleINT8API::setDynamicRange(SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    // populate per tensor dynamic range
    readPerTensorDynamicRangeValues();

    std::cout << "[INFO] Setting Per Tensor Dynamic Range" << std::endl;

    // set dynamic range for for network input tensors
    string name = network->getLayer(0)->getInput(0)->getName();
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        string name = network->getInput(i)->getName();
        network->getInput(i)->setDynamicRange(-mPerTensorDynamicRangeMap.at(name), mPerTensorDynamicRangeMap.at(name));
    }

    // set dynamic range for per layer tensors
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        for (int j = 0; j < network->getLayer(i)->getNbOutputs(); ++j)
        {
            string name = network->getLayer(i)->getOutput(j)->getName();
            if (mPerTensorDynamicRangeMap.find(name) != mPerTensorDynamicRangeMap.end())
            {
                // Calibrator generated dynamic range for network tensor can be overriden or set using below API
                network->getLayer(i)->getOutput(j)->setDynamicRange(-mPerTensorDynamicRangeMap.at(name), mPerTensorDynamicRangeMap.at(name));
            }
        }
    }

    if (mParams.verbose)
    {
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        std::cout << "[INFO] Per Tensor Dynamic Range Values for the Network: " << std::endl;
        for (auto iter = mPerTensorDynamicRangeMap.begin(); iter != mPerTensorDynamicRangeMap.end(); ++iter)
            std::cout << "Tensor: " << iter->first << ". Max Absolute Dynamic Range: " << iter->second << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;
    }
    std::cout << std::endl;
}

//!
//! \brief Preprocess inputs and allocate host/device input buffers
//!
bool sampleINT8API::prepareInput(const samplesCommon::BufferManager& buffers)
{
    std::string input_file = locateFile(mParams.imageFileName, mParams.dataDirs);
    if (samplesCommon::toLower(samplesCommon::getFileType(input_file)).compare("ppm") != 0)
    {
        std::cout << "ERROR: wrong fromat: " << input_file << " is not a ppm file. " << std::endl;
        return false;
    }

    // Prepare PPM Buffer to read the input image
    samplesCommon::PPM<kINPUT_C, kINPUT_H, kINPUT_W> ppm;
    samplesCommon::readPPMFile(input_file, ppm);

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(mInOut["input"]));

    // Convert HWC to CHW and Normalize
    for (int c = 0; c < kINPUT_C; ++c)
    {
        for (int h = 0; h < kINPUT_H; ++h)
        {
            for (int w = 0; w < kINPUT_W; ++w)
            {
                int dstIdx = c * kINPUT_H * kINPUT_W + h * kINPUT_W + w;
                int srcIdx = h * kINPUT_W * kINPUT_C + w * kINPUT_C + c;
                // This equation include 3 steps
                // 1. Scale Image to range [0.f, 1.0f]
                // 2. Normalize Image using per channel Mean and per channel Standard Deviation
                // 3. Shuffle HWC to CHW form
                hostInputBuffer[dstIdx] = (float(ppm.buffer[srcIdx]) / kScale - kMean[c]) / kStdDev[c];
            }
        }
    }
    return true;
}

//!
//! \brief Verifies that the output is correct and prints it
//!
bool sampleINT8API::verifyOutput(const samplesCommon::BufferManager& buffers) const
{
    // copy output host buffer data for further processing
    const float* probPtr = static_cast<const float*>(buffers.getHostBuffer(mInOut.at("output")));
    vector<float> output(probPtr, probPtr + mOutputDims.d[0] * mParams.batchSize);

    auto inds = samplesCommon::argsort(output.cbegin(), output.cend(), true);

    // read reference lables to generate prediction lables
    vector<string> referenceVector;
    if (!samplesCommon::readReferenceFile(locateFile(mParams.referenceFileName, mParams.dataDirs), referenceVector))
    {
        return false;
    }

    vector<string> top5Result = samplesCommon::classify(referenceVector, output, 5);

    std::cout << std::endl
              << "################################" << std::endl;
    std::cout << "sampleINT8API result: Detected: " << std::endl;
    for (int i = 1; i <= 5; ++i)
        std::cout << "[" << i << "]  " << top5Result[i - 1] << std::endl;

    return true;
}

//!
//! \brief This function creates the network, configures the builder and creates the network engine
//!
//! \details This function creates INT8 classification network by parsing the onnx model and builds
//!          the engine that will be used to run INT8 inference (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool sampleINT8API::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
        return false;

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
        return false;

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
        return false;

    // Parse ONNX model file to populate TensorRT INetwork
    int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;
    if (!parser->parseFromFile(locateFile(mParams.modelFileName, mParams.dataDirs).c_str(), verbosity))
    {
        std::cout << "[ERROR] Unable to parse ONNX model file : " << mParams.modelFileName << std::endl;
        return false;
    }

    if (!builder->platformHasFastInt8())
    {
        std::cout << "[ERROR] Platform does not support INT8 Inference. sampleINT8API can only run in INT8 Mode." << std::endl;
        return false;
    }

    samplesCommon::enableDLA(builder.get(), mParams.dlaCore);

    // Configure buider
    builder->allowGPUFallback(true);
    builder->setMaxBatchSize(mParams.batchSize);
    builder->setMaxWorkspaceSize(1_GB);

    // Enable INT8 model. Required to set custom per tensor dynamic range or INT8 Calibration
    builder->setInt8Mode(true);
    // Mark calibrator as null. As user provides dynamic range for each tensor, no calibrator is required
    builder->setInt8Calibrator(nullptr);

    // force layer to execute with required precision
    builder->setStrictTypeConstraints(true);
    setLayerPrecision(network);

    // set INT8 Per Tensor Dynamic range
    setDynamicRange(network);

    // build TRT engine
    mEngine = std::move(std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network), samplesCommon::InferDeleter()));
    if (!mEngine)
        return false;

    // populates input output map structure
    getInputOutputNames();

    // derive input/output dims from engine bindings
    const int inputIndex = mEngine.get()->getBindingIndex(mInOut["input"].c_str());
    mInputDims = mEngine.get()->getBindingDimensions(inputIndex);

    const int outputIndex = mEngine.get()->getBindingIndex(mInOut["output"].c_str());
    mOutputDims = mEngine.get()->getBindingDimensions(outputIndex);

    return true;
}

//!
//! \brief This function runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output
//!
bool sampleINT8API::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
        return false;

    // Read the input data into the managed buffers
    // There should be just 1 input tensor

    if (!prepareInput(buffers))
        return false;

    // Create CUDA stream for the execution of this inference
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    // Asynchronously enqueue the inference work
    if (!context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr))
        return false;

    // Asynchronously copy data from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete
    cudaStreamSynchronize(stream);

    // Release stream
    cudaStreamDestroy(stream);

    // Check and print the output of the inference
    bool outputCorrect = false;
    outputCorrect = verifyOutput(buffers);

    return outputCorrect;
}

//!
//! \brief This function can be used to clean up any state created in the sample class
//!
bool sampleINT8API::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    return true;
}

int main(int argc, char** argv)
{
    SampleINT8APIArgs args;
    parsesampleINT8APIArgs(args, argc, argv);

    if (args.help)
    {
        printHelpInfo();
        return EXIT_FAILURE;
    }

    SampleINT8APIParams params = initializeSampleParams(args);
    sampleINT8API sample(params);
    std::cout << "[INFO] Building and running a INT8 GPU inference engine for " << params.modelFileName << std::endl;

    if (!sample.build())
        return EXIT_FAILURE;
    if (!sample.infer())
        return EXIT_FAILURE;
    if (!sample.teardown())
        return EXIT_FAILURE;

    std::cout << "\nDone." << std::endl;
    return EXIT_SUCCESS;
}
