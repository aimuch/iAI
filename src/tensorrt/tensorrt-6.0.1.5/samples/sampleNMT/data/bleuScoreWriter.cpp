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

#include "bleuScoreWriter.h"
#include "logger.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace nmtSample
{

typedef std::vector<std::string> Segment_t;
typedef std::map<Segment_t, int> Count_t;
int read(std::vector<Segment_t>& samples, std::shared_ptr<std::istream> input, int samplesToRead = 1)
{
    std::string line;
    int lineCounter = 0;
    Segment_t tokens;
    samples.resize(0);
    std::string pattern("@@ ");
    while (lineCounter < samplesToRead && std::getline(*input, line))
    {
        // if clean and handle BPE or SPM outputs is required
        std::size_t p0 = 0;
        while ((p0 = line.find(pattern, p0)) != std::string::npos)
        {
            line.replace(p0, pattern.length(), "");
        }

        // generate error if those special characters exist. Windows needs explicit encoding.
#ifdef _MSC_VER
        p0 = line.find(u8"\u2581");
#else
        p0 = line.find("\u2581");
#endif
        assert((p0 == std::string::npos));
        std::istringstream ss(line);
        std::string token;
        tokens.resize(0);
        while (ss >> token)
        {
            tokens.emplace_back(token);
        }
        samples.emplace_back(tokens);
        lineCounter++;
    }
    return lineCounter;
}

Count_t ngramCounts(const Segment_t& segment, int maxOrder = 4)
{
    Count_t ngramCounts;

    for (int order = 1; order < maxOrder + 1; order++)
    {
        for (int i = 0; i < static_cast<int>(segment.size()) - order + 1; i++)
        {
            Segment_t ngram;
            for (int j = i; j < i + order; j++)
                ngram.emplace_back(segment[j]);

            auto it = ngramCounts.find(ngram);
            if (it != ngramCounts.end())
            {
                it->second++;
            }
            else
                ngramCounts[ngram] = 1;
        }
    }

    return ngramCounts;
}

Count_t ngramCountIntersection(const Count_t& cnt0, const Count_t& cnt1)
{
    Count_t overlap;
    // merge the maps
    auto it0 = cnt0.begin(), it1 = cnt1.begin(), end0 = cnt0.end(), end1 = cnt1.end();
    while (it0 != end0 && it1 != end1)
    {
        if (it0->first == it1->first)
        {
            overlap.emplace(it0->first, std::min(it0->second, it1->second));
            it0++;
            it1++;
        }
        else
        {
            if (it0->first < it1->first)
                it0++;
            else
                it1++;
        }
    }
    return overlap;
}

void accumulateBLEU(const std::vector<Segment_t>& referenceSamples,
                    const std::vector<Segment_t>& outputSamples,
                    int maxOrder,
                    size_t& referenceLength,
                    size_t& translationLength,
                    std::vector<size_t>& matchesByOrder,
                    std::vector<size_t>& possibleMatchesByOrder)
{
    assert(referenceSamples.size() == outputSamples.size());
    auto reference = referenceSamples.begin();
    auto translation = outputSamples.begin();

    while (translation != outputSamples.end())
    {
        referenceLength += reference->size();
        translationLength += translation->size();

        Count_t refNgramCounts = ngramCounts(*reference);
        Count_t outputNgramCounts = ngramCounts(*translation);
        Count_t overlap = ngramCountIntersection(outputNgramCounts, refNgramCounts);
        for (auto& ngram : overlap)
        {
            matchesByOrder[ngram.first.size() - 1] += ngram.second;
        }
        for (int order = 1; order < maxOrder + 1; order++)
        {
            int possibleMatches = static_cast<int>(translation->size()) - order + 1;
            if (possibleMatches > 0)
                possibleMatchesByOrder[order - 1] += possibleMatches;
        }
        ++translation;
        ++reference;
    }
}

BLEUScoreWriter::BLEUScoreWriter(std::shared_ptr<std::istream> referenceTextInput, Vocabulary::ptr vocabulary, int maxOrder)
    : mReferenceInput(referenceTextInput)
    , mVocabulary(vocabulary)
    , mReferenceLength(0)
    , mTranslationLength(0)
    , mMaxOrder(maxOrder)
    , mSmooth(false)
    , mMatchesByOrder(maxOrder, 0)
    , mPossibleMatchesByOrder(maxOrder, 0)
{
}

void BLEUScoreWriter::write(
    const int* hOutputData,
    int actualOutputSequenceLength,
    int actualInputSequenceLength)
{
    std::vector<Segment_t> outputSamples;
    std::vector<Segment_t> referenceSamples;
    int numReferenceSamples = read(referenceSamples, mReferenceInput, 1);
    assert(numReferenceSamples == 1);

    Segment_t segment;
    std::stringstream filteredSentence(DataWriter::generateText(actualOutputSequenceLength, hOutputData, mVocabulary));
    std::string token;
    while (filteredSentence >> token)
    {
        segment.emplace_back(token);
    }
    outputSamples.emplace_back(segment);

    accumulateBLEU(referenceSamples, outputSamples, mMaxOrder, mReferenceLength, mTranslationLength, mMatchesByOrder, mPossibleMatchesByOrder);
}

void BLEUScoreWriter::initialize()
{
}

void BLEUScoreWriter::finalize()
{
    gLogInfo << "BLEU score = " << getScore() << std::endl;
}

float BLEUScoreWriter::getScore() const
{
    std::vector<double> precisions(mMaxOrder, 0.0);
    for (int i = 0; i < mMaxOrder; i++)
    {
        if (mSmooth)
        {
            precisions[i] = ((mMatchesByOrder[i] + 1.) / (mPossibleMatchesByOrder[i] + 1.));
        }
        else
        {
            if (mPossibleMatchesByOrder[i] > 0)
                precisions[i] = (static_cast<double>(mMatchesByOrder[i]) / mPossibleMatchesByOrder[i]);
            else
                precisions[i] = 0.0;
        }
    }
    double pLogSum, geoMean;
    if (*std::min_element(precisions.begin(), precisions.end()) > 0.0)
    {
        pLogSum = 0.0;
        for (auto p : precisions)
            pLogSum += (1. / mMaxOrder) * log(p);
        geoMean = exp(pLogSum);
    }
    else
        geoMean = 0.0;

    double ratio = static_cast<double>(mTranslationLength) / mReferenceLength;
    double bp;
    bp = (ratio > 1.0) ? 1.0 : exp(1.0 - 1.0 / ratio);
    return static_cast<float>(geoMean * bp * 100.0);
}

std::string BLEUScoreWriter::getInfo()
{
    std::stringstream ss;
    ss << "BLEU Score Writer, max order = " << mMaxOrder;
    return ss.str();
}
} // namespace nmtSample
