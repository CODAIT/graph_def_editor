# Copyright 2019 IBM. All Rights Reserved.
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
Tests for rewrite.py in the GraphDef Editor
"""

import shutil
import tempfile
import unittest
import tensorflow as tf
import numpy as np

import graph_def_editor as gde


class RewriteTest(unittest.TestCase):

  def test_change_batch_size(self):
    """Basic test for gde.rewrite.change_batch_size."""
    tf_g = tf.Graph()
    with tf_g.as_default():
      input_tensor = tf.placeholder(dtype=tf.int32, shape=[32, 1],
                                    name="Input")
      result_tensor = input_tensor + 42
    g = gde.Graph(tf_g)
    gde.rewrite.change_batch_size(g, 3, [g[input_tensor.op.name]])

    with g.to_tf_graph().as_default():
      with tf.Session() as sess:
        result = sess.run(result_tensor.name,
                          {input_tensor.name:
                           np.array([0, 1, 2]).reshape([3, 1])})
        self.assertTrue(np.array_equal(result,
                                       np.array([42, 43, 44]).reshape([3, 1])))

  def test_change_batch_size_variable_size(self):
    """
    Verifies that a batch size of None (variable size) works.
    Also verifies that passing a tensor instead of node works.
    """
    tf_g = tf.Graph()
    with tf_g.as_default():
      input_tensor = tf.placeholder(dtype=tf.float32, shape=[32, 1],
                                    name="Input")
      result_tensor = input_tensor + 42.0
    g = gde.Graph(tf_g)
    # Note that we pass a Tensor as the third argument instead of a Node.
    gde.rewrite.change_batch_size(g, None, [g[input_tensor.name]])

    with g.to_tf_graph().as_default():
      with tf.Session() as sess:
        result = sess.run(result_tensor.name,
                          {input_tensor.name:
                           np.array([0]).reshape([1, 1])})
        self.assertTrue(np.array_equal(result,
                                       np.array([42.]).reshape([1, 1])))
        result = sess.run(result_tensor.name,
                          {input_tensor.name:
                           np.array([0, 1]).reshape([2, 1])})
        self.assertTrue(np.array_equal(result,
                                       np.array([42., 43.]).reshape([2, 1])))

  def test_change_batch_size_saved_model(self):
    """
    Verifies that changes of batch size survive serializing the graph as a
    SavedModel
    """
    temp_dir = tempfile.mkdtemp()
    try:
      tf_g = tf.Graph()
      with tf_g.as_default():
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[32, 1],
                                      name="Input")
        result_tensor = input_tensor + 42.0
        with tf.Session() as sess:
          tf.saved_model.simple_save(sess, temp_dir + "/model_before",
                                     inputs={"in": input_tensor},
                                     outputs={"out": result_tensor})

      # Make sure the original SavedModel loads properly
      with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING],
                            temp_dir + "/model_before")

      g = gde.saved_model_to_graph(temp_dir + "/model_before")
      gde.rewrite.change_batch_size(g, None, [g[input_tensor.name]])
      g.to_saved_model(temp_dir + "/model_after")

      with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING],
                            temp_dir + "/model_after")
        result = sess.run(result_tensor.name,
                          {input_tensor.name:
                           np.array([0]).reshape([1, 1])})
        self.assertTrue(np.array_equal(result,
                                       np.array([42.]).reshape([1, 1])))
        result = sess.run(result_tensor.name,
                          {input_tensor.name:
                           np.array([0, 1]).reshape([2, 1])})
        self.assertTrue(np.array_equal(result,
                                       np.array([42., 43.]).reshape([2, 1])))
    finally:
      # Remove temp dir unconditionally. Comment out try and finally if you
      # want the directory to stick around after a test failure.
      shutil.rmtree(temp_dir)
