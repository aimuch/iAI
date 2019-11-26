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

#include "softmaxLikelihood.h"

#include <cassert>

#include <math.h>

namespace nmtSample
{
void SoftmaxLikelihood::addToModel(
    nvinfer1::INetworkDefinition* network,
    int beamWidth,
    nvinfer1::ITensor* inputLogits,
    nvinfer1::ITensor* inputLikelihoods,
    nvinfer1::ITensor** newCombinedLikelihoods,
    nvinfer1::ITensor** newRayOptionIndices,
    nvinfer1::ITensor** newVocabularyIndices)
{
    auto softmaxLayer = network->addSoftMax(*inputLogits);
    assert(softmaxLayer != nullptr);
    softmaxLayer->setName("Softmax in likelihood calculation");
    softmaxLayer->setAxes(2);
    auto softmaxTensor = softmaxLayer->getOutput(0);
    assert(softmaxTensor != nullptr);

    auto topKLayer = network->addTopK(*softmaxTensor, nvinfer1::TopKOperation::kMAX, beamWidth, 2);
    assert(topKLayer != nullptr);
    topKLayer->setName("TopK 1st in likelihood calculation");
    auto newLikelihoods = topKLayer->getOutput(0);
    assert(newLikelihoods != nullptr);
    auto vocabularyIndices = topKLayer->getOutput(1);
    assert(vocabularyIndices != nullptr);

    auto eltWiseLayer = network->addElementWise(*newLikelihoods, *inputLikelihoods, nvinfer1::ElementWiseOperation::kPROD);
    assert(eltWiseLayer != nullptr);
    eltWiseLayer->setName("EltWise multiplication in likelihood calculation");
    auto combinedLikelihoods = eltWiseLayer->getOutput(0);
    assert(combinedLikelihoods != nullptr);

    auto shuffleLayer = network->addShuffle(*combinedLikelihoods);
    assert(shuffleLayer != nullptr);
    shuffleLayer->setName("Reshape combined likelihoods");
    nvinfer1::Dims shuffleDims{1, {beamWidth * beamWidth}, {nvinfer1::DimensionType::kCHANNEL}};
    shuffleLayer->setReshapeDimensions(shuffleDims);
    auto reshapedCombinedLikelihoods = shuffleLayer->getOutput(0);
    assert(reshapedCombinedLikelihoods != nullptr);

    auto topKLayer2 = network->addTopK(*reshapedCombinedLikelihoods, nvinfer1::TopKOperation::kMAX, beamWidth, 1);
    assert(topKLayer2 != nullptr);
    topKLayer2->setName("TopK 2nd in likelihood calculation");
    *newCombinedLikelihoods = topKLayer2->getOutput(0);
    assert(*newCombinedLikelihoods != nullptr);
    *newRayOptionIndices = topKLayer2->getOutput(1);
    assert(*newRayOptionIndices != nullptr);

    auto shuffleLayer2 = network->addShuffle(*vocabularyIndices);
    assert(shuffleLayer2 != nullptr);
    shuffleLayer2->setName("Reshape vocabulary indices");
    nvinfer1::Dims shuffleDims2{1, {beamWidth * beamWidth}, {nvinfer1::DimensionType::kCHANNEL}};
    shuffleLayer2->setReshapeDimensions(shuffleDims2);
    auto reshapedVocabularyIndices = shuffleLayer2->getOutput(0);
    assert(reshapedVocabularyIndices != nullptr);

    auto gatherLayer = network->addGather(*reshapedVocabularyIndices, **newRayOptionIndices, 0);
    assert(gatherLayer != nullptr);
    gatherLayer->setName("Shuffle vocabulary indices");
    *newVocabularyIndices = gatherLayer->getOutput(0);
    assert(*newVocabularyIndices != nullptr);
}

float SoftmaxLikelihood::SoftmaxLikelihoodCombinationOperator::combine(float rayLikelihood, float optionLikelihood) const
{
    return rayLikelihood * optionLikelihood;
}

float SoftmaxLikelihood::SoftmaxLikelihoodCombinationOperator::init() const
{
    return 1.0F;
}

float SoftmaxLikelihood::SoftmaxLikelihoodCombinationOperator::smallerThanMinimalLikelihood() const
{
    return -1.0F;
}

LikelihoodCombinationOperator::ptr SoftmaxLikelihood::getLikelihoodCombinationOperator() const
{
    return std::make_shared<SoftmaxLikelihoodCombinationOperator>();
}

std::string SoftmaxLikelihood::getInfo()
{
    return "Softmax Likelihood";
}
} // namespace nmtSample
