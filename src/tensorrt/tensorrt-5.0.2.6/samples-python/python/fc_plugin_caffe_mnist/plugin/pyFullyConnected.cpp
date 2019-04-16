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

    // Since the createPlugin function overrides IPluginFactory functionality, we do not need to explicitly bind it here.
    // We specify py::multiple_inheritance because we have not explicitly specified nvinfer1::IPluginFactory as a base class.
    py::class_<FCPluginFactory, nvcaffeparser1::IPluginFactoryExt>(m, "FCPluginFactory", py::multiple_inheritance())
        // Bind the default constructor.
        .def(py::init<>())
        // The destroy_plugin function does not override the base class, so we must bind it explicitly.
        .def("destroy_plugin", &FCPluginFactory::destroyPlugin)
    ;
}
