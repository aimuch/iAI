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

# This file contains functions for training a TensorFlow model
import tensorflow as tf
import numpy as np
import os

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))

MODEL_DIR = os.path.join(
    WORKING_DIR,
    'models'
)


def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.reshape(x_train, (-1, 1, 28, 28))
    x_test = np.reshape(x_test, (-1, 1, 28, 28))
    return x_train, y_train, x_test, y_test

def build_model():
    # Create the keras model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[1, 28, 28], name="InputLayer"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation(activation=tf.nn.relu6, name="ReLU6"))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="OutputLayer"))
    return model

def train_model():
    # Build and compile model
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Train the model on the data
    model.fit(
        x_train, y_train,
        epochs = 10,
        verbose = 1
    )

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test loss: {}\nTest accuracy: {}".format(test_loss, test_acc))

    return model

def maybe_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_model(model):
    output_names = model.output.op.name
    sess = tf.keras.backend.get_session()

    graphdef = sess.graph.as_graph_def()
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, [output_names])
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    # Make directory to save model in if it doesn't exist already
    maybe_mkdir(MODEL_DIR)

    model_path = os.path.join(MODEL_DIR, "trained_lenet5.pb")
    with open(model_path, "wb") as ofile:
        ofile.write(frozen_graph.SerializeToString())


if __name__ == "__main__":
    model = train_model()
    save_model(model)
