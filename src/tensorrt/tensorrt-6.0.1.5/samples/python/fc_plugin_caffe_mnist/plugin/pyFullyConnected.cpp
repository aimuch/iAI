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

#include "FullyConnected.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(fcplugin, m)
{
    namespace py = pybind11;

    // This allows us to use the bindings exposed by the tensorrt module.
    py::module::import("tensorrt");

    // Note that we only need to bind the constructors manually. Since all other methods override IPlugin functionality, they will be automatically available in the python bindings.
    // The `std::unique_ptr<FCPlugin, py::nodelete>` specifies that Python is not responsible for destroying the object. This is required because the destructor is private.
    py::class_<FCPlugin, nvinfer1::IPluginExt, std::unique_ptr<FCPlugin, py::nodelete>>(m, "FCPlugin")
        // Bind the normal constructor as well as the one which deserializes the plugin
        .def(py::init<const nvinfer1::Weights*, int>())
        .def(py::init<const void*, size_t>())
    ;

    // Our custom plugin factory derives from both nvcaffeparser1::IPluginFactoryExt and nvinfer1::IPluginFactory
    py::class_<FCPluginFactory, nvcaffeparser1::IPluginFactoryExt, nvinfer1::IPluginFactory>(m, "FCPluginFactory")
        // Bind the default constructor.
        .def(py::init<>())
        // The destroy_plugin function does not override either of the base classes, so we must bind it explicitly.
        .def("destroy_plugin", &FCPluginFactory::destroyPlugin)
    ;
}
