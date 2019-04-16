//!
//! sampleGoogleNet.cpp
//! This file contains the implementation of the GoogleNet sample. It creates the network using
//! the GoogleNet caffe model.
//! It can be run with the following command line:
//! Command: ./sample_googlenet [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

#include "common.h"
#include "argsParser.h"
#include "buffers.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>

static Logger gLogger;

//!
//! \brief  The SampleGoogleNet class implements the GoogleNet sample
//!
//! \details It creates the network using a caffe model
//!
class SampleGoogleNet
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleGoogleNet(const samplesCommon::CaffeSampleParams& params)
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

    samplesCommon::CaffeSampleParams mParams;

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr; //!< The TensorRT engine used to run the network

    //!
    //! \brief This function parses a Caffe model for GoogleNet and creates a TensorRT network
    //!
    void constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser);
};

//!
//! \brief This function creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the GoogleNet network by parsing the caffe model and builds
//!          the engine that will be used to run GoogleNet (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleGoogleNet::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
        return false;

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
        return false;

    auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    if (!parser)
        return false;

    constructNetwork(builder, network, parser);

    mEngine = std::move(std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network), samplesCommon::InferDeleter()));
    if (!mEngine)
        return false;

    return true;
}

//!
//! \brief This function uses a caffe parser to create the googlenet Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the googlenet network
//!
//! \param builder Pointer to the engine builder
//!
void SampleGoogleNet::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser)
{
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
        locateFile(mParams.prototxtFileName, mParams.dataDirs).c_str(),
        locateFile(mParams.weightsFileName, mParams.dataDirs).c_str(),
        *network,
        nvinfer1::DataType::kFLOAT);

    for (auto& s : mParams.outputTensorNames)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    builder->setMaxBatchSize(mParams.batchSize);
    builder->setMaxWorkspaceSize(16_MB);
    samplesCommon::enableDLA(builder.get(), mParams.dlaCore);
}

//!
//! \brief This function runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleGoogleNet::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
        return false;

    // Fetch host buffers and set host input buffers to all zeros
    for (auto& input : mParams.inputTensorNames)
        memset(buffers.getHostBuffer(input), 0, buffers.size(input));

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    if (!status)
        return false;

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    return true;
}

//!
//! \brief This function can be used to clean up any state created in the sample class
//!
bool SampleGoogleNet::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief This function initializes members of the params struct using the command line args
//!
samplesCommon::CaffeSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::CaffeSampleParams params;
    if (args.dataDirs.size() != 0) //!< Use the data directory provided by the user
        params.dataDirs = args.dataDirs;
    else //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/googlenet/");
        params.dataDirs.push_back("data/samples/googlenet/");
    }
    params.prototxtFileName = "googlenet.prototxt";
    params.weightsFileName = "googlenet.caffemodel";
    params.inputTensorNames.push_back("data");
    params.batchSize = 4;
    params.outputTensorNames.push_back("prob");
    params.dlaCore = args.useDLACore;

    return params;
}
//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_googlenet [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use data/samples/googlenet/ and data/googlenet/" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;

    if (!samplesCommon::parseArgs(args, argc, argv))
    {
        if (args.help)
        {
            printHelpInfo();
            return EXIT_SUCCESS;
        }
        return EXIT_FAILURE;
    }

    samplesCommon::CaffeSampleParams params = initializeSampleParams(args);
    SampleGoogleNet sample(params);
    std::cout << "Building and running a GPU inference engine for GoogleNet" << std::endl;

    if (!sample.build())
        return EXIT_FAILURE;
    if (!sample.infer())
        return EXIT_FAILURE;
    if (!sample.teardown())
        return EXIT_FAILURE;

    std::cout << "Ran " << argv[0] << " with \n"
              << "Input(s): ";
    for (auto& input : sample.mParams.inputTensorNames)
        std::cout << input << " ";
    std::cout << "\nOutput(s): ";
    for (auto& output : sample.mParams.outputTensorNames)
        std::cout << output << " ";

    std::cout << "\nDone." << std::endl;
    return EXIT_SUCCESS;
}
