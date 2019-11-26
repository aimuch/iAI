#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

# Model download and UFF convertion utils
import os
import sys
import tarfile

import requests
import tensorflow as tf
import tensorrt as trt
import graphsurgeon as gs
import uff
import time
import math

from utils.paths import PATHS


# UFF conversion functionality

# This class contains converted (UFF) model metadata
class ModelData(object):
    # Name of input node
    INPUT_NAME = "Input"
    # CHW format of model input
    INPUT_SHAPE = (3, 300, 300)
    # Name of output node
    OUTPUT_NAME = "NMS"

    @staticmethod
    def get_input_channels():
        return ModelData.INPUT_SHAPE[0]

    @staticmethod
    def get_input_height():
        return ModelData.INPUT_SHAPE[1]

    @staticmethod
    def get_input_width():
        return ModelData.INPUT_SHAPE[2]


def ssd_unsupported_nodes_to_plugin_nodes(ssd_graph):
    """Makes ssd_graph TensorRT comparible using graphsurgeon.

    This function takes ssd_graph, which contains graphsurgeon
    DynamicGraph data structure. This structure describes frozen Tensorflow
    graph, that can be modified using graphsurgeon (by deleting, adding,
    replacing certain nodes). The graph is modified by removing
    Tensorflow operations that are not supported by TensorRT's UffParser
    and replacing them with custom layer plugin nodes.

    Note: This specific implementation works only for
    ssd_inception_v2_coco_2017_11_17 network.

    Args:
        ssd_graph (gs.DynamicGraph): graph to convert
    Returns:
        gs.DynamicGraph: UffParser compatible SSD graph
    """
    # Create TRT plugin nodes to replace unsupported ops in Tensorflow graph
    channels = ModelData.get_input_channels()
    height = ModelData.get_input_height()
    width = ModelData.get_input_width()

    Input = gs.create_plugin_node(name="Input",
        op="Placeholder",
        dtype=tf.float32,
        shape=[1, channels, height, width])
    PriorBox = gs.create_plugin_node(name="GridAnchor", op="GridAnchor_TRT",
        minSize=0.2,
        maxSize=0.95,
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1,0.1,0.2,0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6
    )
    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=1e-8,
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=91,
        inputOrder=[0, 2, 1],
        confSigmoid=1,
        isNormalized=1
    )
    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        dtype=tf.float32,
        axis=2
    )
    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
    )
    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
    )

    # Create a mapping of namespace names -> plugin nodes.
    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "ToFloat": Input,
        "image_tensor": Input,
        "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
        "MultipleGridAnchorGenerator/Identity": concat_priorbox,
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }

    # Create a new graph by collapsing namespaces
    ssd_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (NMS).
    # If remove_exclusive_dependencies is True, the whole graph will be removed!
    ssd_graph.remove(ssd_graph.graph_outputs, remove_exclusive_dependencies=False)
    return ssd_graph

def model_to_uff(model_path, output_uff_path, silent=False):
    """Takes frozen .pb graph, converts it to .uff and saves it to file.

    Args:
        model_path (str): .pb model path
        output_uff_path (str): .uff path where the UFF file will be saved
        silent (bool): if False, writes progress messages to stdout

    """
    dynamic_graph = gs.DynamicGraph(model_path)
    dynamic_graph = ssd_unsupported_nodes_to_plugin_nodes(dynamic_graph)

    uff.from_tensorflow(
        dynamic_graph.as_graph_def(),
        [ModelData.OUTPUT_NAME],
        output_filename=output_uff_path,
        text=True
    )


# Model download functionality

def maybe_print(should_print, print_arg):
    """Prints message if supplied boolean flag is true.

    Args:
        should_print (bool): if True, will print print_arg to stdout
        print_arg (str): message to print to stdout
    """
    if should_print:
        print(print_arg)

def maybe_mkdir(dir_path):
    """Makes directory if it doesn't exist.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def download_file(file_url, file_dest_path, silent=False):
    """Downloads file from supplied URL and puts it into supplied directory.

    Args:
        file_url (str): URL with file to download
        file_dest_path (str): path to save downloaded file in
        silent (bool): if False, writes progress messages to stdout
    """
    with open(file_dest_path, "wb") as f:
        maybe_print(not silent, "Downloading {}".format(file_dest_path))
        response = requests.get(file_url, stream=True)
        total_length = response.headers.get('content-length')

        def print_progress(pct_done):
            isatty = sys.stdout.isatty()
            clear_char = "\r" if isatty else ""
            endl_char = "" if isatty else "\n"
            progress_bar_width = int(math.floor(pct_done * 50 / 100.0))
            sys.stdout.write("{}Download progress [{}{}] {:.2f}%{}".format(
                  clear_char,
                  "=" * progress_bar_width,
                  " " * (50 - progress_bar_width),
                  pct_done,
                  endl_char))
            sys.stdout.flush()

        if total_length is None or silent: # no content length header or silent, just write file
            f.write(response.content)
        else: # not silent, print progress
            dl = 0
            total_length = int(total_length)
            t_last_update = t_cur = time.time()
            for data in response.iter_content(chunk_size=(4096 * 1024)):
                dl += len(data)
                f.write(data)
                if t_cur - t_last_update > 2.0:
                    print_progress(100 * dl / total_length)
                    t_last_update = t_cur
                t_cur = time.time()
            print_progress(100)
            sys.stdout.write("\n")

def download_model(model_name, silent=False):
    """Downloads model_name from Tensorflow model zoo.

    Args:
        model_name (str): chosen object detection model
        silent (bool): if False, writes progress messages to stdout
    """
    maybe_print(not silent, "Preparing pretrained model")
    model_dir = PATHS.get_models_dir_path()
    maybe_mkdir(model_dir)
    model_url = PATHS.get_model_url(model_name)
    model_archive_path = os.path.join(model_dir, "{}.tar.gz".format(model_name))
    download_file(model_url, model_archive_path, silent)
    maybe_print(not silent, "Download complete\nUnpacking {}".format(model_archive_path))
    with tarfile.open(model_archive_path, "r:gz") as tar:
        tar.extractall(path=model_dir)
    maybe_print(not silent, "Extracting complete\nRemoving {}".format(model_archive_path))
    os.remove(model_archive_path)
    maybe_print(not silent, "Model ready")

def prepare_ssd_model(model_name="ssd_inception_v2_coco_2017_11_17", silent=False):
    """Downloads pretrained object detection model and converts it to UFF.

    The model is downloaded from Tensorflow object detection model zoo.
    Currently only ssd_inception_v2_coco_2017_11_17 model is supported
    due to model_to_uff() using logic specific to that network when converting.

    Args:
        model_name (str): chosen object detection model
        silent (bool): if False, writes progress messages to stdout
    """
    if model_name != "ssd_inception_v2_coco_2017_11_17":
        raise NotImplementedError(
            "Model {} is not supported yet".format(model_name))
    download_model(model_name, silent)
    ssd_pb_path = PATHS.get_model_pb_path(model_name)
    ssd_uff_path = PATHS.get_model_uff_path(model_name)
    model_to_uff(ssd_pb_path, ssd_uff_path, silent)
