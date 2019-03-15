# -*- coding: utf-8 -*-
# Author : Andy Liu
# Last modified: 2019-03-15

# This script is used to convert tensorflow model file to uff file
# Using: 
#        python tf_to_uff.py

import uff
import tensorflow as tf
import tensorrt as trt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ckpt_path = "model/model.ckpt"
forzen_model_path = "model/frozen_graphs/frozen_graph.pb"
uff_path = "model/uff/model.uff"


frozen_input_name = "input"
net_input_shape = (3, 32, 32)
frozen_output_names = ["fc_3/frozen"]

def getChatBotModel(ckpt_path):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(ckpt_path+'.meta')
        saver.restore(sess, ckpt_path)
        graph = tf.get_default_graph().as_graph_def()
        #graph = tf.get_default_graph()
        #print('graph list:', graph.get_operations())
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph, frozen_output_names)
        return tf.graph_util.remove_training_nodes(frozen_graph)


tf_model = getChatBotModel(ckpt_path)
with tf.gfile.FastGFile(forzen_model_path, mode='wb') as f:
        f.write(tf_model.SerializeToString())
#uff_model = uff.from_tensorflow(tf_model, output_nodes=frozen_output_names, output_filename=uff_path, text=True)
uff_model = uff.from_tensorflow_frozen_model(forzen_model_path, output_nodes=frozen_output_names, output_filename=uff_path, text=True)
print('Success! UFF file is in ', os.path.abspath(uff_path))
