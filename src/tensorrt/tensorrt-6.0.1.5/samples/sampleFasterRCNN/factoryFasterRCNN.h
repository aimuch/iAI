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

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

const int poolingH = 7;
const int poolingW = 7;
const int featureStride = 16;
const int preNmsTop = 6000;
const int nmsMaxOut = 300;
const int anchorsRatioCount = 3;
const int anchorsScaleCount = 3;
const float iouThreshold = 0.7f;
const float minBoxSize = 16;
const float spatialScale = 0.0625f;
const float anchorsRatios[anchorsRatioCount] = {0.5f, 1.0f, 2.0f};
const float anchorsScales[anchorsScaleCount] = {8.0f, 16.0f, 32.0f};

class FRCNNPluginFactory : public nvcaffeparser1::IPluginFactoryV2
{
public:
    virtual nvinfer1::IPluginV2* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights, const char* libNamespace) override
    {
        assert(isPluginV2(layerName));
        if (!strcmp(layerName, "RPROIFused"))
        {
            assert(mPluginRPROI == nullptr);
            assert(nbWeights == 0 && weights == nullptr);
            mPluginRPROI = std::unique_ptr<IPluginV2, decltype(pluginDeleter)>(createRPNROIPlugin(featureStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale,
                                                                                                  DimsHW(poolingH, poolingW), Weights{nvinfer1::DataType::kFLOAT, anchorsRatios, anchorsRatioCount},
                                                                                                  Weights{nvinfer1::DataType::kFLOAT, anchorsScales, anchorsScaleCount}),
                                                                               pluginDeleter);
            mPluginRPROI.get()->setPluginNamespace(libNamespace);
            return mPluginRPROI.get();
        }
        else
        {
            assert(0);
            return nullptr;
        }
    }

    // caffe parser plugin implementation
    bool isPluginV2(const char* name) override { return !strcmp(name, "RPROIFused"); }

    void destroyPlugin()
    {
        mPluginRPROI.reset();
    }

    void (*pluginDeleter)(IPluginV2*){[](IPluginV2* ptr) { ptr->destroy(); }};
    std::unique_ptr<IPluginV2, decltype(pluginDeleter)> mPluginRPROI{nullptr, pluginDeleter};
};
