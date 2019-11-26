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
#ifndef ERROR_RECORDER_H
#define ERROR_RECORDER_H
#include <vector>
#include <mutex>
#include <cstdint>
#include <atomic>
#include <exception>
#include "NvInferRuntimeCommon.h"
using namespace nvinfer1;
//!
//! A simple imeplementation of the IErrorRecorder interface for
//! use by samples. This interface also can be used as a reference
//! implementation.
//! The sample Error recorder is based on a vector that pairs the error
//! code and the error string into a single element. It also uses
//! standard mutex's and atomics in order to make sure that the code
//! works in a multi-threaded environment.
//! SampleErrorRecorder is not intended for use in automotive safety
//! environments.
//!
class SampleErrorRecorder : public IErrorRecorder
{
    using errorPair = std::pair<ErrorCode, std::string>;
    using errorStack = std::vector<errorPair>;

    public:
        SampleErrorRecorder() = default;

        virtual ~SampleErrorRecorder() noexcept {}
        int32_t getNbErrors() const noexcept final
        {
            return mErrorStack.size();
        }
        ErrorCode getErrorCode(int32_t errorIdx) const noexcept final
        {
            return indexCheck(errorIdx) ? ErrorCode::kINVALID_ARGUMENT : (*this)[errorIdx].first;
        };
        IErrorRecorder::ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept final
        {
            return indexCheck(errorIdx) ? "errorIdx out of range." : (*this)[errorIdx].second.c_str();
        }
        // This class can never overflow since we have dynamic resize via std::vector usage.
        bool hasOverflowed() const noexcept final
        {
            return false;
        }

        // Empty the errorStack.
        void clear() noexcept final
        {
            try 
            {
                // grab a lock so that there is no addition while clearing.
                std::lock_guard<std::mutex> guard(mStackLock);
                mErrorStack.clear();
            }
            catch (const std::exception& e)
            {
                getLogger()->log(ILogger::Severity::kINTERNAL_ERROR, e.what());
            }
        };

        //! Simple helper function that 
        bool empty() const noexcept
        {
            return mErrorStack.empty();
        }

        bool reportError(ErrorCode val, IErrorRecorder::ErrorDesc desc) noexcept final {
            try
            {
                std::lock_guard<std::mutex> guard(mStackLock);
                mErrorStack.push_back(errorPair(val, desc));
            }
            catch(const std::exception& e)
            {
                getLogger()->log(ILogger::Severity::kINTERNAL_ERROR, e.what());
            }
            // All errors are considered fatal.
            return true;
        }

        // Atomically increment or decrement the ref counter.
        IErrorRecorder::RefCount incRefCount() noexcept final
        {
            return ++mRefCount;
        }
        IErrorRecorder::RefCount decRefCount() noexcept final
        {
            return --mRefCount;
        }

    private:
        // Simple helper functions.
        const errorPair& operator[](size_t index) const noexcept
        {
            return mErrorStack[index];
        }

        bool indexCheck(int32_t index) const noexcept
        {
            // By converting signed to unsigned, we only need a single check since
            // negative numbers turn into large positive greater than the size.
            size_t sIndex = index;
            return sIndex >= mErrorStack.size();
        }
        // Mutex to hold when locking mErrorStack.
        std::mutex mStackLock;

        // Reference count of the class. Destruction of the class when mRefCount
        // is not zero causes undefined behavior.
        std::atomic<int32_t> mRefCount{0};

        // The error stack that holds the errors recorded by TensorRT.
        errorStack mErrorStack;
}; // class SampleErrorRecorder
#endif // ERROR_RECORDER_H
