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

#ifndef SAMPLE_NMT_ENCODER_
#define SAMPLE_NMT_ENCODER_

#include <memory>
#include <vector>

#include "../component.h"
#include "NvInfer.h"

namespace nmtSample
{
/** \class Encoder
    *
    * \brief encodes input sentences into output states
    *
    */
class Encoder : public Component
{
public:
    typedef std::shared_ptr<Encoder> ptr;

    Encoder() = default;

    /**
        * \brief add the memory and last timestep states to the network
        * lastTimestepHiddenStates is the pointer to the tensor where the encoder stores all layer hidden states for the last timestep (which is dependent on the sample),
        * the function should define the tensor, it could be nullptr indicating these data are not needed
        */
    virtual void addToModel(
        nvinfer1::INetworkDefinition* network,
        int maxInputSequenceLength,
        nvinfer1::ITensor* inputEmbeddedData,
        nvinfer1::ITensor* actualInputSequenceLengths,
        nvinfer1::ITensor** inputStates,
        nvinfer1::ITensor** memoryStates,
        nvinfer1::ITensor** lastTimestepStates)
        = 0;

    /**
        * \brief get the size of the memory state vector
        */
    virtual int getMemoryStatesSize() = 0;

    /**
        * \brief get the sizes (vector of them) of the hidden state vectors
        */
    virtual std::vector<nvinfer1::Dims> getStateSizes() = 0;

    ~Encoder() override = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_ENCODER_
