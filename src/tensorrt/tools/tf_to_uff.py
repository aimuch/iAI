# -*- coding: utf-8 -*-
# Author : Andy Liu
# Last modified: 2019-03-15

# This script is used to convert tensorflow model file to uff file
# Using: 
#        python tf_to_uff.py

import os
import uff
import tensorflow as tf
import tensorrt as trt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_path = "model/model.ckpt"
frozen_model_path = "model/frozen_graphs/frozen_graph.pb"
uff_path = "model/uff/model.uff"
frozen_node_name = ["fc_3/frozen"]

def getFrozenModel(model_path):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(model_path+'.meta')
        saver.restore(sess, model_path)
        graph = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph, frozen_node_name)
        return tf.graph_util.remove_training_nodes(frozen_graph)


tf_model = getFrozenModel(model_path)
with tf.gfile.FastGFile(frozen_model_path, mode='wb') as f:
        f.write(tf_model.SerializeToString())
#uff_model = uff.from_tensorflow(tf_model, output_nodes=frozen_node_name, output_filename=uff_path, text=True)
uff_model = uff.from_tensorflow_frozen_model(frozen_model_path, output_nodes=frozen_node_name, output_filename=uff_path, text=True)

print('Success! Frozen model is stored in ', os.path.abspath(frozen_model_path))
print('Success! UFF file is stored in ', os.path.abspath(uff_path))
