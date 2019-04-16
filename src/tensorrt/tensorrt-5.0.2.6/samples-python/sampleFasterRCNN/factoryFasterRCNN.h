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
