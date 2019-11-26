/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

//!
//! sampleFasterRCNN.cpp
//! This file contains the implementation of the FasterRCNN sample. It creates the network using
//! the FasterRCNN caffe model.
//! It can be run with the following command line:
//! Command: ./sample_fasterRCNN [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "factoryFasterRCNN.h"

const std::string gSampleName = "TensorRT.sample_fasterRCNN";

//!
//! \brief The SampleFasterRCNNParams structure groups the additional parameters required by
//!         the FasterRCNN sample.
//!
struct SampleFasterRCNNParams : public samplesCommon::CaffeSampleParams
{
    int outputClsSize; //!< The number of output classes
    int nmsMaxOut;     //!< The maximum number of detection post-NMS
};

//! \brief  The SampleFasterRCNN class implements the FasterRCNN sample
//!
//! \details It creates the network using a caffe model
//!
class SampleFasterRCNN
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleFasterRCNN(const SampleFasterRCNNParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    SampleFasterRCNNParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    static const int kIMG_CHANNELS = 3;
    static const int kIMG_H = 375;
    static const int kIMG_W = 500;
    std::vector<samplesCommon::PPM<kIMG_CHANNELS, kIMG_H, kIMG_W>> mPPMs; //!< PPMs of test images

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses a Caffe model for FasterRCNN and creates a TensorRT network
    //!
    void constructNetwork(SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser,
        SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
        SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Filters output detections, handles post-processing of bounding boxes and verify results
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Performs inverse bounding box transform and clipping
    //!
    void bboxTransformInvAndClip(const float* rois, const float* deltas, float* predBBoxes, const float* imInfo,
        const int N, const int nmsMaxOut, const int numCls);

    //!
    //! \brief Performs non maximum suppression on final bounding boxes
    //!
    std::vector<int> nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex, float* bbox,
        const int classNum, const int numClasses, const float nmsThreshold);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the FasterRCNN network by parsing the caffe model and builds
//!          the engine that will be used to run FasterRCNN (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleFasterRCNN::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    if (!parser)
    {
        return false;
    }

    FRCNNPluginFactory pluginFactory;
    parser->setPluginFactoryV2(&pluginFactory);
    constructNetwork(parser, builder, network, config);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    assert(network->getNbInputs() == 2);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    pluginFactory.destroyPlugin();

    return true;
}

//!
//! \brief Uses a caffe parser to create the FasterRCNN network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the FasterRCNN network
//!
//! \param builder Pointer to the engine builder
//!
void SampleFasterRCNN::constructNetwork(SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser,
    SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
    SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor
        = parser->parse(locateFile(mParams.prototxtFileName, mParams.dataDirs).c_str(),
            locateFile(mParams.weightsFileName, mParams.dataDirs).c_str(), *network, nvinfer1::DataType::kFLOAT);

    for (auto& s : mParams.outputTensorNames)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16_MiB);
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleFasterRCNN::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 2);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Post-process detections and verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool SampleFasterRCNN::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool SampleFasterRCNN::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputC = mInputDims.d[0];
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
    const int batchSize = mParams.batchSize;

    // Available images
    const std::vector<std::string> imageList = {"000456.ppm", "000542.ppm", "001150.ppm", "001763.ppm", "004545.ppm"};
    mPPMs.resize(batchSize);
    assert(mPPMs.size() <= imageList.size());

    // Fill im_info buffer
    float* hostImInfoBuffer = static_cast<float*>(buffers.getHostBuffer("im_info"));
    for (int i = 0; i < batchSize; ++i)
    {
        readPPMFile(locateFile(imageList[i], mParams.dataDirs), mPPMs[i]);
        hostImInfoBuffer[i * 3] = float(mPPMs[i].h);     // Number of rows
        hostImInfoBuffer[i * 3 + 1] = float(mPPMs[i].w); // Number of columns
        hostImInfoBuffer[i * 3 + 2] = 1;                 // Image scale
    }

    // Fill data buffer
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("data"));
    // Pixel mean used by the Faster R-CNN's author
    const float pixelMean[3]{102.9801f, 115.9465f, 122.7717f}; // Also in BGR order
    for (int i = 0, volImg = inputC * inputH * inputW; i < batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            // The color image to input should be in BGR order
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
                hostDataBuffer[i * volImg + c * volChl + j] = float(mPPMs[i].buffer[j * inputC + 2 - c]) - pixelMean[c];
        }
    }

    return true;
}

//!
//! \brief Filters output detections and handles post-processing of bounding boxes, verify result
//!
//! \return whether the detection output matches expectations
//!
bool SampleFasterRCNN::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int batchSize = mParams.batchSize;
    const int nmsMaxOut = mParams.nmsMaxOut;
    const int outputClsSize = mParams.outputClsSize;
    const int outputBBoxSize = mParams.outputClsSize * 4;

    const float* imInfo = static_cast<const float*>(buffers.getHostBuffer("im_info"));
    const float* deltas = static_cast<const float*>(buffers.getHostBuffer("bbox_pred"));
    const float* clsProbs = static_cast<const float*>(buffers.getHostBuffer("cls_prob"));
    float* rois = static_cast<float*>(buffers.getHostBuffer("rois"));

    // Unscale back to raw image space
    for (int i = 0; i < batchSize; ++i)
    {
        for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
        {
            rois[i * nmsMaxOut * 4 + j] /= imInfo[i * 3 + 2];
        }
    }

    std::vector<float> predBBoxes(batchSize * nmsMaxOut * outputBBoxSize, 0);
    bboxTransformInvAndClip(rois, deltas, predBBoxes.data(), imInfo, batchSize, nmsMaxOut, outputClsSize);

    const float nmsThreshold = 0.3f;
    const float score_threshold = 0.8f;
    const std::vector<std::string> classes{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"};

    // The sample passes if there is at least one detection for each item in the batch
    bool pass = true;

    for (int i = 0; i < batchSize; ++i)
    {
        float* bbox = predBBoxes.data() + i * nmsMaxOut * outputBBoxSize;
        const float* scores = clsProbs + i * nmsMaxOut * outputClsSize;
        int numDetections = 0;
        for (int c = 1; c < outputClsSize; ++c) // Skip the background
        {
            std::vector<std::pair<float, int>> scoreIndex;
            for (int r = 0; r < nmsMaxOut; ++r)
            {
                if (scores[r * outputClsSize + c] > score_threshold)
                {
                    scoreIndex.push_back(std::make_pair(scores[r * outputClsSize + c], r));
                    std::stable_sort(scoreIndex.begin(), scoreIndex.end(),
                        [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {
                            return pair1.first > pair2.first;
                        });
                }
            }

            // Apply NMS algorithm
            const std::vector<int> indices = nonMaximumSuppression(scoreIndex, bbox, c, outputClsSize, nmsThreshold);

            numDetections += static_cast<int>(indices.size());

            // Show results
            for (unsigned k = 0; k < indices.size(); ++k)
            {
                const int idx = indices[k];
                const std::string storeName
                    = classes[c] + "-" + std::to_string(scores[idx * outputClsSize + c]) + ".ppm";
                gLogInfo << "Detected " << classes[c] << " in " << mPPMs[i].fileName << " with confidence "
                         << scores[idx * outputClsSize + c] * 100.0f << "% "
                         << " (Result stored in " << storeName << ")." << std::endl;

                const samplesCommon::BBox b{bbox[idx * outputBBoxSize + c * 4], bbox[idx * outputBBoxSize + c * 4 + 1],
                    bbox[idx * outputBBoxSize + c * 4 + 2], bbox[idx * outputBBoxSize + c * 4 + 3]};
                writePPMFileWithBBox(storeName, mPPMs[i], b);
            }
        }
        pass &= numDetections >= 1;
    }

    return pass;
}

//!
//! \brief Performs inverse bounding box transform
//!
void SampleFasterRCNN::bboxTransformInvAndClip(const float* rois, const float* deltas, float* predBBoxes,
    const float* imInfo, const int N, const int nmsMaxOut, const int numCls)
{
    for (int i = 0; i < N * nmsMaxOut; ++i)
    {
        float width = rois[i * 4 + 2] - rois[i * 4] + 1;
        float height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
        float ctr_x = rois[i * 4] + 0.5f * width;
        float ctr_y = rois[i * 4 + 1] + 0.5f * height;
        const float* imInfo_offset = imInfo + i / nmsMaxOut * 3;
        for (int j = 0; j < numCls; ++j)
        {
            float dx = deltas[i * numCls * 4 + j * 4];
            float dy = deltas[i * numCls * 4 + j * 4 + 1];
            float dw = deltas[i * numCls * 4 + j * 4 + 2];
            float dh = deltas[i * numCls * 4 + j * 4 + 3];
            float pred_ctr_x = dx * width + ctr_x;
            float pred_ctr_y = dy * height + ctr_y;
            float pred_w = exp(dw) * width;
            float pred_h = exp(dh) * height;
            predBBoxes[i * numCls * 4 + j * 4]
                = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 1]
                = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 2]
                = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
            predBBoxes[i * numCls * 4 + j * 4 + 3]
                = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
        }
    }
}

//!
//! \brief Performs non maximum suppression on final bounding boxes
//!
std::vector<int> SampleFasterRCNN::nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex, float* bbox,
    const int classNum, const int numClasses, const float nmsThreshold)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
        float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
        float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
        float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
        float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::vector<int> indices;
    for (auto i : scoreIndex)
    {
        const int idx = i.second;
        bool keep = true;
        for (unsigned k = 0; k < indices.size(); ++k)
        {
            if (keep)
            {
                const int kept_idx = indices[k];
                float overlap = computeIoU(
                    &bbox[(idx * numClasses + classNum) * 4], &bbox[(kept_idx * numClasses + classNum) * 4]);
                keep = overlap <= nmsThreshold;
            }
            else
            {
                break;
            }
        }
        if (keep)
        {
            indices.push_back(idx);
        }
    }
    return indices;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleFasterRCNNParams initializeSampleParams(const samplesCommon::Args& args)
{
    SampleFasterRCNNParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/faster-rcnn/");
        params.dataDirs.push_back("data/samples/faster-rcnn/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.prototxtFileName = "faster_rcnn_test_iplugin.prototxt";
    params.weightsFileName = "VGG16_faster_rcnn_final.caffemodel";
    params.inputTensorNames.push_back("data");
    params.inputTensorNames.push_back("im_info");
    params.batchSize = 5;
    params.outputTensorNames.push_back("bbox_pred");
    params.outputTensorNames.push_back("cls_prob");
    params.outputTensorNames.push_back("rois");
    params.dlaCore = args.useDLACore;

    params.outputClsSize = 21;
    params.nmsMaxOut
        = 300; // This value needs to be changed as per the nmsMaxOut value set in RPROI plugin parameters in prototxt

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_fasterRCNN [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/faster-rcnn/ and data/faster-rcnn/"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleFasterRCNN sample(initializeSampleParams(args));

    gLogInfo << "Building and running a GPU inference engine for FasterRCNN" << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
