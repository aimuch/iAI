import graphsurgeon as gs
import tensorflow as tf

def preprocess(dynamic_graph):
    axis = dynamic_graph.find_nodes_by_path("concatenate/concat/axis")[0]
    # Set axis to 2, because of discrepancies between TensorFlow and TensorRT.
    axis.attr["value"].tensor.int_val[0] = 2
