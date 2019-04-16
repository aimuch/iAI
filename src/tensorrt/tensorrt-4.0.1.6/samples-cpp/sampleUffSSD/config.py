import graphsurgeon as gs
import tensorflow as tf

Input = gs.create_node("Input",
    op="Placeholder",
    dtype=tf.float32,
    shape=[1, 3, 300, 300])
PriorBox = gs.create_node("PriorBox",
    numLayers=6,
    minScale=0.2,
    maxScale=0.95,
    aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
    layerVariances=[0.1,0.1,0.2,0.2],
    featureMapShapes=[19, 10, 5, 3, 2, 1])
NMS = gs.create_node("NMS",
    scoreThreshold=1e-8,
    iouThreshold=0.6,
    maxDetectionsPerClass=100,
    maxTotalDetections=100,
    numClasses=91,
    scoreConverter="SIGMOID")
concat_priorbox = gs.create_node("concat_priorbox", dtype=tf.float32, axis=2)
concat_box_loc = gs.create_node("concat_box_loc")
concat_box_conf = gs.create_node("concat_box_conf")

namespace_plugin_map = {
    "MultipleGridAnchorGenerator": PriorBox,
    "Postprocessor": NMS,
    "Preprocessor": Input,
    "ToFloat": Input,
    "image_tensor": Input,
    "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,
    "concat": concat_box_loc,
    "concat_1": concat_box_conf
}

def preprocess(dynamic_graph):
    # Now create a new graph by collapsing namespaces
    dynamic_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (NMS).
    dynamic_graph.remove(dynamic_graph.graph_outputs, remove_exclusive_dependencies=False)
