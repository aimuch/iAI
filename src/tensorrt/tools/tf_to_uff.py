import uff
import tensorflow as tf
import tensorrt as trt
import os

filepath = "model/model.ckpt"
forzen_model_path = "model/frozen_graphs/frozen_graph.pb"
output_path = "model/uff/model.uff"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def getChatBotModel(filepath):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(filepath+'.meta')
        saver.restore(sess, filepath)
        graph = tf.get_default_graph().as_graph_def()
        #graph = tf.get_default_graph()
        #print('graph list:', graph.get_operations())
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph, ["fc_3/frozen"])
        return tf.graph_util.remove_training_nodes(frozen_graph)


tf_model = getChatBotModel(filepath)
with tf.gfile.FastGFile(forzen_model_path, mode='wb') as f:
        f.write(tf_model.SerializeToString())
#uff_model = uff.from_tensorflow(tf_model, List_nodes=["lanenet_loss/instance_seg", "lanenet_loss/binary_seg"], output_filename=output_path, text=True)
uff_model = uff.from_tensorflow_frozen_model(forzen_model_path, output_nodes=["fc_3/frozen"], output_filename=output_path, text=True)
print('Done！')
