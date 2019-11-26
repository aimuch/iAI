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

#include "beamSearchPolicy.h"
#ifdef _MSC_VER
//Macro definition needed to avoid name collision with std::min/max and Windows.h min/max
#define NOMINMAX
#endif
#include <algorithm>
#include <cassert>
#include <limits>
#include <sstream>

namespace nmtSample
{
BeamSearchPolicy::BeamSearchPolicy(
    int endSequenceId,
    LikelihoodCombinationOperator::ptr likelihoodCombinationOperator,
    int beamWidth)
    : mEndSequenceId(endSequenceId)
    , mLikelihoodCombinationOperator(likelihoodCombinationOperator)
    , mBeamWidth(beamWidth)
{
}

void BeamSearchPolicy::initialize(
    int sampleCount,
    int* maxOutputSequenceLengths)
{
    mSampleCount = sampleCount;
    mMaxOutputSequenceLengths.resize(mSampleCount);
    std::copy(maxOutputSequenceLengths, maxOutputSequenceLengths + mSampleCount, &mMaxOutputSequenceLengths[0]);

    mValidSamples.resize(mSampleCount);
    std::fill(mValidSamples.begin(), mValidSamples.end(), true);

    mCurrentLikelihoods.resize(mSampleCount * mBeamWidth);
    std::fill(mCurrentLikelihoods.begin(), mCurrentLikelihoods.end(), mLikelihoodCombinationOperator->init());

    mBeamSearchTable.clear();

    mTimestepId = 0;

    mCandidates.resize(mSampleCount);
    mCandidateLikelihoods.resize(mSampleCount);
    std::fill(mCandidateLikelihoods.begin(), mCandidateLikelihoods.end(), mLikelihoodCombinationOperator->smallerThanMinimalLikelihood());
}

void BeamSearchPolicy::processTimestep(
    int validSampleCount,
    const float* hCombinedLikelihoods,
    const int* hVocabularyIndices,
    const int* hRayOptionIndices,
    int* hSourceRayIndices,
    float* hSourceLikelihoods)
{
    ++mTimestepId;
    mBeamSearchTable.resize(mTimestepId * mSampleCount * mBeamWidth);
    auto baseBeamSearchTable = mBeamSearchTable.begin() + (mTimestepId - 1) * mSampleCount * mBeamWidth;

    for (int sampleId = 0; sampleId < validSampleCount; ++sampleId)
    {
        auto currentSourceRayIndices = hSourceRayIndices + sampleId * mBeamWidth;
        auto currentLikelihoods = hSourceLikelihoods + sampleId * mBeamWidth;
        auto currentBeamSearchTable = baseBeamSearchTable + sampleId * mBeamWidth;

        int rayId = 0;
        if (mValidSamples[sampleId])
        {
            for (; rayId < mBeamWidth; ++rayId)
            {
                float optionCombinedLikelihood = hCombinedLikelihoods[sampleId * mBeamWidth + rayId];

                // Check if the current candidate is already better than this option
                if (optionCombinedLikelihood <= mCandidateLikelihoods[sampleId])
                    break; // The remaining options are even worse

                int optionOriginalRayId = hRayOptionIndices[sampleId * mBeamWidth + rayId] / mBeamWidth;
                int optionVocabularyId = hVocabularyIndices[sampleId * mBeamWidth + rayId];

                if ((optionVocabularyId == mEndSequenceId) || (mTimestepId >= mMaxOutputSequenceLengths[sampleId]))
                {
                    // We have a new candidate output sequence for the sample
                    mCandidateLikelihoods[sampleId] = optionCombinedLikelihood;
                    auto& candidate = mCandidates[sampleId];
                    candidate.resize(mTimestepId);
                    backtrack(mTimestepId - 2, sampleId, optionOriginalRayId, &candidate[0], mTimestepId - 2);
                    candidate[mTimestepId - 1] = optionVocabularyId;
                    break;
                }

                *(currentSourceRayIndices + rayId) = optionOriginalRayId;
                *(currentLikelihoods + rayId) = optionCombinedLikelihood;
                (currentBeamSearchTable + rayId)->vocabularyId = optionVocabularyId;
                (currentBeamSearchTable + rayId)->backtrackId = optionOriginalRayId;
            }

            // No valid rays left for the sample
            if (rayId == 0)
                mValidSamples[sampleId] = false;
        }

        // Mark the remaining rays as invalid ones
        for (; rayId < mBeamWidth; ++rayId)
        {
            *(currentSourceRayIndices + rayId) = 0;
            *(currentLikelihoods + rayId) = mLikelihoodCombinationOperator->smallerThanMinimalLikelihood();
            (currentBeamSearchTable + rayId)->vocabularyId = mEndSequenceId;
            (currentBeamSearchTable + rayId)->backtrackId = 0;
        }
    }
}

int BeamSearchPolicy::getTailWithNoWorkRemaining()
{
    for (int sampleId = mSampleCount - 1; sampleId >= 0; --sampleId)
    {
        if (mValidSamples[sampleId])
            return sampleId + 1;
    }
    return 0;
}

void BeamSearchPolicy::readGeneratedResult(
    int sampleCount,
    int maxOutputSequenceLength,
    int* hOutputData,
    int* hActualOutputSequenceLengths)
{
    for (int sampleId = 0; sampleId < sampleCount; ++sampleId)
    {
        if (mCandidateLikelihoods[sampleId] > mLikelihoodCombinationOperator->smallerThanMinimalLikelihood())
        {
            // We have a candidate (finished sequence)
            std::copy_n(
                mCandidates[sampleId].begin(),
                std::min(static_cast<int>(mCandidates[sampleId].size()), maxOutputSequenceLength),
                hOutputData + sampleId * maxOutputSequenceLength);
            hActualOutputSequenceLengths[sampleId] = mCandidates[sampleId].size();
        }
        else
        {
            // We don't have a finished sequence generated, will output the unfinished one with the highest likelihood
            assert(mValidSamples[sampleId]);
            backtrack(mTimestepId - 1, sampleId, 0, hOutputData + sampleId * maxOutputSequenceLength, maxOutputSequenceLength - 1);
            hActualOutputSequenceLengths[sampleId] = mTimestepId;
        }
    }
}

void BeamSearchPolicy::backtrack(
    int lastTimestepId,
    int sampleId,
    int lastTimestepRayId,
    int* hOutputData,
    int lastTimestepWriteId) const
{
    int rayId = lastTimestepRayId;
    for (int timestepId = lastTimestepId; timestepId >= 0; --timestepId)
    {
        const auto& entry = mBeamSearchTable[(timestepId * mSampleCount + sampleId) * mBeamWidth + rayId];
        rayId = entry.backtrackId;
        if (timestepId <= lastTimestepWriteId)
            hOutputData[timestepId] = entry.vocabularyId;
    }
}

std::string BeamSearchPolicy::getInfo()
{
    std::stringstream ss;
    ss << "Beam Search Policy, beam = " << mBeamWidth;
    return ss.str();
}
} // namespace nmtSample
