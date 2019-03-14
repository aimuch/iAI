/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
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

#ifndef _FULLY_CONNECTED_H_
#define _FULLY_CONNECTED_H_

#include <assert.h>
#include <iostream>
#include <sys/stat.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <string.h>
#include <cstdint>
#include <cstring>

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        std::cout << "Cuda failure: " << status;        \
        abort();                                        \
    }                                                   \
}

class FCPlugin: public IPlugin
{
public:
	FCPlugin(const Weights *weights, int nbWeights)
	{
		// in this simple case we're going to infer the number of output channels from the bias weights
		// the knowledge that the kernel weights are weights[0] and the bias weights are weights[1] was
		// divined from the caffe innards

		assert(nbWeights == 2);
		mKernelWeights = copyToDevice(weights[0].values, weights[0].count);
		mBiasWeights = copyToDevice(weights[1].values, weights[1].count);
	}

	// create the plugin at runtime from a byte stream
	FCPlugin(const void* data, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(data), *a = d;
		mKernelWeights = copyToDevice(d+sizeof(int), *reinterpret_cast<const int*>(d));
		d += sizeof(int) + mKernelWeights.count * sizeof(float);
		mBiasWeights = copyToDevice(d + sizeof(int), *reinterpret_cast<const int*>(d));
		d += sizeof(int) + mBiasWeights.count * sizeof(float);
		assert(d == a + length);
	}

	~FCPlugin()
	{
		cudaFree(const_cast<void*>(mKernelWeights.values));
		cudaFree(const_cast<void*>(mBiasWeights.values));
	}

	int getNbOutputs() const override
	{
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
		return DimsCHW(mBiasWeights.count, 1, 1);
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
	{
		assert(nbInputs == 1 && inputDims[0].d[1] == 1 && inputDims[0].d[2] == 1);
		assert(nbOutputs == 1 && outputDims[0].d[1] == 1 && outputDims[0].d[2] == 1);
		assert(mKernelWeights.count == inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2] * mBiasWeights.count);
	}

	int initialize() override
	{
		CHECK(cudnnCreate(&mCudnn));							// initialize cudnn and cublas
		CHECK(cublasCreate(&mCublas));
		CHECK(cudnnCreateTensorDescriptor(&mSrcDescriptor));	// create cudnn tensor descriptors we need for bias addition
		CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));

		return 0;
	}

	virtual void terminate() override
	{
		CHECK(cublasDestroy(mCublas));
		CHECK(cudnnDestroy(mCudnn));
	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
	{
		int nbOutputChannels = mBiasWeights.count;
		int nbInputChannels = mKernelWeights.count / nbOutputChannels;
		float kONE = 1.0f, kZERO = 0.0f;
		cublasSetStream(mCublas, stream);
		cudnnSetStream(mCudnn, stream);
		CHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, nbOutputChannels, batchSize, nbInputChannels, &kONE, 
				reinterpret_cast<const float*>(mKernelWeights.values), nbInputChannels, 
				reinterpret_cast<const float*>(inputs[0]), nbInputChannels, &kZERO, 
				reinterpret_cast<float*>(outputs[0]), nbOutputChannels));
		CHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, nbOutputChannels, 1, 1));
		CHECK(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, nbOutputChannels, 1, 1));
		CHECK(cudnnAddTensor(mCudnn, &kONE, mSrcDescriptor, mBiasWeights.values, &kONE, mDstDescriptor, outputs[0]));
		return 0;
	}

	virtual size_t getSerializationSize() override
	{
		return sizeof(int) * 2 + mKernelWeights.count * sizeof(float) + mBiasWeights.count*sizeof(float);
	}

	virtual void serialize(void* buffer) override
	{
		char* d = reinterpret_cast<char*>(buffer), *a = d;
		d += copyFromDevice(d, mKernelWeights);
		d += copyFromDevice(d, mBiasWeights);
		assert(d == a + getSerializationSize());
	}
private:

	Weights copyToDevice(const void* hostData, int count)
	{
		void* deviceData;
		CHECK(cudaMalloc(&deviceData, count * sizeof(float)));
		CHECK(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
		return Weights{ DataType::kFLOAT, deviceData, count };
	}

	int copyFromDevice(char* hostBuffer, Weights deviceWeights)
	{
		*reinterpret_cast<int*>(hostBuffer) = deviceWeights.count;
		cudaMemcpy(hostBuffer + sizeof(int), deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
		return sizeof(int) + deviceWeights.count * sizeof(float);
	}

	cudnnHandle_t mCudnn;
	cublasHandle_t mCublas;
	Weights mKernelWeights, mBiasWeights;
	cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
};

// integration for serialization
class FullyConnectedPluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
	// caffe parser plugin implementation
	bool isPlugin(const char* name) override
	{
		return !strcmp(name, "ip2");
	}

	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
		assert(isPlugin(layerName) && nbWeights == 2 && weights[0].type == DataType::kFLOAT && weights[1].type == DataType::kFLOAT);
		assert(mPlugin == nullptr);
		mPlugin = new FCPlugin(weights, nbWeights);
		return mPlugin;
	}

	// deserialization plugin implementation
	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{		
		assert(isPlugin(layerName));
		assert(mPlugin == nullptr);
		mPlugin = new FCPlugin(serialData, serialLength);
		return mPlugin;
	}

	// the application has to destroy the plugin when it knows it's safe to do so
	void destroyPlugin()
	{
		delete mPlugin;
	}

	FCPlugin* mPlugin{ nullptr };
};

#endif //_FULLY_CONNECTED_H
