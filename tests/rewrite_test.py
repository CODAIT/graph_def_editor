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

import unittest
import tensorflow as tf
import numpy as np

import graph_def_editor as gde


class RewriteTest(unittest.TestCase):

  def test_change_batch_size(self):
    """Basic test for gde.rewrite.change_batch_size."""
    tf_g = tf.Graph()
    with tf_g.as_default():
      input_tensor = tf.placeholder(dtype=tf.int32, shape=[32,1],
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
      input_tensor = tf.placeholder(dtype=tf.float32, shape=[32,1],
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

