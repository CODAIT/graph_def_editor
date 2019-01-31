# Coypright 2018 IBM. All Rights Reserved.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Example of using the GraphDef editor to adjust the batch size of a pretrained
model.

To run this example from the root of the project, type:
   PYTHONPATH=$PWD env/bin/python examples/batch_size_example.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import graph_def_editor as gde
import shutil
import tarfile
import textwrap
import urllib.request

FLAGS = tf.flags.FLAGS


def _indent(s):
  return textwrap.indent(str(s), "    ")


_TMP_DIR = "/tmp/batch_size_example"
_MODEL_URL = "http://download.tensorflow.org/models/official/20181001_resnet" \
             "/savedmodels/resnet_v2_fp16_savedmodel_NHWC.tar.gz"
_MODEL_TARBALL = _TMP_DIR + "/resnet_v2_fp16_savedmodel_NHWC.tar.gz"
_SAVED_MODEL_DIR = _TMP_DIR + "/resnet_v2_fp16_savedmodel_NHWC/1538686978"
_AFTER_MODEL_DIR = _TMP_DIR + "/rewritten_model"


def main(_):
  # Grab a copy of the official TensorFlow ResNet50 model in fp16.
  # See https://github.com/tensorflow/models/tree/master/official/resnet
  # Cache the tarball so we don't download it repeatedly
  if not os.path.isdir(_SAVED_MODEL_DIR):
    if os.path.isdir(_TMP_DIR):
      shutil.rmtree(_TMP_DIR)
    os.mkdir(_TMP_DIR)
    print("Downloading model tarball from {}".format(_MODEL_URL))
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_TARBALL)
    print("Unpacking SavedModel from {} to {}".format(_MODEL_TARBALL, _TMP_DIR))
    with tarfile.open(_MODEL_TARBALL) as t:
      t.extractall(_TMP_DIR)

  # Load the SavedModel
  tf_g = tf.Graph()
  with tf.Session(graph=tf_g) as sess:
    tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING],
                        _SAVED_MODEL_DIR)

  # print("Graph is:\n{}".format(tf_g.as_graph_def()))

  # Print out some statistics about tensor shapes
  print("BEFORE:")
  print("  Input tensor is {}".format(tf_g.get_tensor_by_name(
    "input_tensor:0")))
  print("  Softmax tensor is {}".format(tf_g.get_tensor_by_name(
    "softmax_tensor:0")))

  # Convert the SavedModel to a gde.Graph and rewrite the batch size to None
  g = gde.saved_model_to_graph(_SAVED_MODEL_DIR)
  gde.rewrite.change_batch_size(g, new_size=None, inputs=[g["input_tensor"]])
  if os.path.exists(_AFTER_MODEL_DIR):
    shutil.rmtree(_AFTER_MODEL_DIR)
  g.to_saved_model(_AFTER_MODEL_DIR)

  # Load the rewritten SavedModel into a TensorFlow graph
  after_tf_g = tf.Graph()
  with tf.Session(graph=after_tf_g) as sess:
    tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING],
                        _AFTER_MODEL_DIR)
    print("AFTER:")
    print("  Input tensor is {}".format(after_tf_g.get_tensor_by_name(
      "input_tensor:0")))
    print("  Softmax tensor is {}".format(after_tf_g.get_tensor_by_name(
      "softmax_tensor:0")))

    # Feed a single array of zeros through the graph
    print("Running inference on dummy data")
    result = sess.run("softmax_tensor:0",
                      {"input_tensor:0": np.zeros([1, 224, 224, 3])})
    print("Result is {}".format(result))


if __name__ == "__main__":
  tf.app.run()
