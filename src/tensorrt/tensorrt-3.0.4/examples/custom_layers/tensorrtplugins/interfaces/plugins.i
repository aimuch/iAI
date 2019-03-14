/*
 * plugin.i
 * Interface file for generating Python wraper for
 * the IPlugin and IPluginFactory classes
 */

%module plugins
%{
#define SWIG_FILE_WITH_INIT
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "src/FullyConnected.h"
//ADD OTHER PLUGINS HERE:
%}

%import "NvInfer.h"
%rename (caffePlugin) nvcaffeparser1::IPluginFactory;
%import "NvCaffeParser.h"
%import "NvInferPlugin.h"

%include "src/FullyConnected.h"
//ADD OTHER PLUGINS HERE:

