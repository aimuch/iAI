#ifndef __GIE_HPP__
#define __GIE_HPP__

//#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <cstring>
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "ModelPara.hpp"
//#include <dw/dnn/DNN.h>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

#define OUTPUT_DIM 1

class GieModel
{
public:

    GieModel(struct ModelPara*);

    ~GieModel() ;

    inline bool isLoaded()
    {
        return m_success;
    }

    void reset();
    //bool inferSingleFrame(const dwImageCUDA *const frame, bool doClustering);
	bool inferSingleFrame(void);
	ICudaEngine* createEngine(std::string engineFile);

	float *m_hostOutputData[OUTPUT_DIM];
	float *m_deviceInputData;
	struct ModelPara para;
	//dwContextHandle_t dwContext;
//	dwRect roi; /* region of interested */

protected:

private:
    uint32_t m_networkInputSize;
    uint32_t m_networkOutputSize[OUTPUT_DIM];
    float *m_deviceOutputData[OUTPUT_DIM];

    bool m_success;

	ICudaEngine* engine;
	IExecutionContext *context;
	void *buffers[2];
	cudaStream_t stream;
	//dwDataConditionerParams *pDataCondiPara;
	//dwDataConditionerHandle_t m_dataConditionerHandle;
	void printEngineInfo();
	int ParseParaFromEngine();
	int initDataCondi(void);
};

#endif // __GIE_HPP__
