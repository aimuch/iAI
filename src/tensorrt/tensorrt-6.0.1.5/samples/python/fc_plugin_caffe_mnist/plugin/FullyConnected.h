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

#ifndef _FULLY_CONNECTED_H_
#define _FULLY_CONNECTED_H_

#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>

#include "NvInfer.h"
#include "NvCaffeParser.h"

#define CHECK(status) { if (status != 0) throw std::runtime_error(__FILE__ +  __LINE__ + std::string{"CUDA Error: "} + std::to_string(status)); }

// Helpers to move data to/from the GPU.
nvinfer1::Weights copyToDevice(const void* hostData, int count)
{
	void* deviceData;
	CHECK(cudaMalloc(&deviceData, count * sizeof(float)));
	CHECK(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
	return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, deviceData, count};
}

int copyFromDevice(char* hostBuffer, nvinfer1::Weights deviceWeights)
{
	*reinterpret_cast<int*>(hostBuffer) = deviceWeights.count;
	CHECK(cudaMemcpy(hostBuffer + sizeof(int), deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost));
	return sizeof(int) + deviceWeights.count * sizeof(float);
}

class FCPlugin: public nvinfer1::IPluginExt
{
public:
	// In this simple case we're going to infer the number of output channels from the bias weights.
	// The knowledge that the kernel weights are weights[0] and the bias weights are weights[1] was
	// divined from the caffe innards
	FCPlugin(const nvinfer1::Weights* weights, int nbWeights)
	{
		assert(nbWeights == 2);
		mKernelWeights = copyToDevice(weights[0].values, weights[0].count);
		mBiasWeights = copyToDevice(weights[1].values, weights[1].count);
	}

	// Create the plugin at runtime from a byte stream.
	FCPlugin(const void* data, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(data);
		const char* check = d;
		// Deserialize kernel.
		const int kernelCount = reinterpret_cast<const int*>(d)[0];
		mKernelWeights = copyToDevice(d + sizeof(int), kernelCount);
		d += sizeof(int) + mKernelWeights.count * sizeof(float);
		// Deserialize bias.
		const int biasCount = reinterpret_cast<const int*>(d)[0];
		mBiasWeights = copyToDevice(d + sizeof(int), biasCount);
		d += sizeof(int) + mBiasWeights.count * sizeof(float);
		// Check that the sizes are what we expected.
		assert(d == check + length);
	}

	virtual int getNbOutputs() const override { return 1; }

	virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override
	{
		assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
		return nvinfer1::DimsCHW{static_cast<int>(mBiasWeights.count), 1, 1};
	}

	virtual int initialize() override
	{
		CHECK(cudnnCreate(&mCudnn));
		CHECK(cublasCreate(&mCublas));
		// Create cudnn tensor descriptors for bias addition.
		CHECK(cudnnCreateTensorDescriptor(&mSrcDescriptor));
		CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));
		return 0;
	}

	virtual void terminate() override
	{
		CHECK(cudnnDestroyTensorDescriptor(mSrcDescriptor));
		CHECK(cudnnDestroyTensorDescriptor(mDstDescriptor));
		CHECK(cublasDestroy(mCublas));
		CHECK(cudnnDestroy(mCudnn));
	}

    // This plugin requires no workspace memory during build time.
	virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

	virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override
	{
		int nbOutputChannels = mBiasWeights.count;
		int nbInputChannels = mKernelWeights.count / nbOutputChannels;
		constexpr float kONE = 1.0f, kZERO = 0.0f;
		// Do matrix multiplication.
		cublasSetStream(mCublas, stream);
		cudnnSetStream(mCudnn, stream);
		CHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, nbOutputChannels, batchSize, nbInputChannels, &kONE,
				reinterpret_cast<const float*>(mKernelWeights.values), nbInputChannels,
				reinterpret_cast<const float*>(inputs[0]), nbInputChannels, &kZERO,
				reinterpret_cast<float*>(outputs[0]), nbOutputChannels));
        // Add bias.
		CHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, nbOutputChannels, 1, 1));
		CHECK(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, nbOutputChannels, 1, 1));
		CHECK(cudnnAddTensor(mCudnn, &kONE, mSrcDescriptor, mBiasWeights.values, &kONE, mDstDescriptor, outputs[0]));
		return 0;
	}

	// For this sample, we'll only support float32 with NCHW.
	virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override
	{
		return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW);
	}

	void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
	{
		assert(nbInputs == 1 && inputDims[0].d[1] == 1 && inputDims[0].d[2] == 1);
		assert(nbOutputs == 1 && outputDims[0].d[1] == 1 && outputDims[0].d[2] == 1);
		assert(mKernelWeights.count == inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2] * mBiasWeights.count);
	}

	virtual size_t getSerializationSize() override
	{
		return sizeof(int) * 2 + mKernelWeights.count * sizeof(float) + mBiasWeights.count * sizeof(float);
	}

	virtual void serialize(void* buffer) override
	{
		char* d = reinterpret_cast<char*>(buffer);
		const char* check = d;
		d += copyFromDevice(d, mKernelWeights);
		d += copyFromDevice(d, mBiasWeights);
		assert(d == check + getSerializationSize());
	}

	// Free buffers.
	virtual ~FCPlugin()
	{
		cudaFree(const_cast<void*>(mKernelWeights.values));
		mKernelWeights.values = nullptr;
		cudaFree(const_cast<void*>(mBiasWeights.values));
		mBiasWeights.values = nullptr;
	}

private:
	cudnnHandle_t mCudnn;
	cublasHandle_t mCublas;
	nvinfer1::Weights mKernelWeights{nvinfer1::DataType::kFLOAT, nullptr}, mBiasWeights{nvinfer1::DataType::kFLOAT, nullptr};
	cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
};

class FCPluginFactory : public nvcaffeparser1::IPluginFactoryExt, public nvinfer1::IPluginFactory
{
public:
	bool isPlugin(const char* name) override { return isPluginExt(name); }

	bool isPluginExt(const char* name) override { return !strcmp(name, "ip2"); }

    // Create a plugin using provided weights.
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
        try
        {
		    assert(isPluginExt(layerName) && nbWeights == 2);
		    assert(mPlugin == nullptr);
            // This plugin will need to be manually destroyed after parsing the network, by calling destroyPlugin.
		    mPlugin = new FCPlugin{weights, nbWeights};
		    return mPlugin;
        }
        catch (std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }

        return nullptr;
	}

    // Create a plugin from serialized data.
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
		try
		{
			assert(isPlugin(layerName));
        	// This will be automatically destroyed when the engine is destroyed.
			return new FCPlugin{serialData, serialLength};
		}
		catch (std::exception& e)
		{
			std::cerr << e.what() << std::endl;
		}

		return nullptr;
	}

    // User application destroys plugin when it is safe to do so.
    // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
	void destroyPlugin() { delete mPlugin; }

    FCPlugin* mPlugin{ nullptr };
};

#endif //_FULLY_CONNECTED_H
