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

  def assertClose(self, expected: np.ndarray, actual: np.ndarray, delta: float):
    """
    Assert that all values in two arrays are within a certain distance of
    each other.
    """
    abs_diff = np.abs(expected - actual)
    max_diff_flat_ix = np.argmax(abs_diff)
    max_diff_ix = np.unravel_index(max_diff_flat_ix, abs_diff.shape)
    max_diff = abs_diff[max_diff_ix]
    self.failIf(max_diff > delta,
                msg="Maximum difference of {} at index {} is greater than "
                    "tolerance {}".format(max_diff, max_diff_ix, delta))

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

  def test_fold_batch_norms_conv_2d(self):
    """
    Python port of TestFoldBatchNormsConv2D in the TF Graph Transform Tool
    tests.
    """
    input_data = (
      np.array([1., 4., 2., 5., 3., 6., -1., -4., -2., -5., -3., -6.],
               dtype=np.float32).reshape([1, 1, 6, 2])
    )
    weights_data = (
      np.array([1., 2., 3., 4., 0.1, 0.2, 0.3, 0.4],
               dtype=np.float32).reshape([1, 2, 2, 2])
    )
    mul_values_data = (
      np.array([2., 3.], dtype=np.float32).reshape([2])
    )

    # Create and run graph:
    # (input, weights) --> Conv2D --> Mul(const)
    tf_g = tf.Graph()
    with tf_g.as_default():
      in_t = tf.constant(input_data, name="input_op")
      weights_t = tf.constant(weights_data, name="weights_op")
      conv_t = tf.nn.conv2d(in_t, weights_t, [1, 1, 1, 1], "VALID",
                            name="conv_op")
      mul_values_t = tf.constant(mul_values_data, name="mul_values")
      output_t = tf.multiply(conv_t, mul_values_t, name="output")
    with tf.Session(graph=tf_g) as sess:
      original_outputs = sess.run(output_t)

    # Rewrite and compare results
    g = gde.Graph(tf_g)
    gde.rewrite.fold_batch_norms(g)
    print("After:\n{}".format(g.to_graph_def()))
    with tf.Session(graph=g.to_tf_graph()) as sess:
      fused_outputs = sess.run(output_t.name)

    self.assertClose(original_outputs, fused_outputs, delta=1e-5)

    # Make sure the rewrite happened
    for n in g.nodes:
      self.assertNotEqual(n.op_type, "Mul")

  def test_fold_old_batch_norms(self):
    """
    Python port of TestFoldOldBatchNorms() in the TF Graph Transfrom Tool
    tests.
    """
    input_data = (
      np.array([1., 4., 2., 5., 3., 6., -1., -4., -2., -5., -3., -6.],
               dtype=np.float32).reshape([1, 1, 6, 2])
    )
    weights_data = (
      np.array([1., 2., 3., 4., 0.1, 0.2, 0.3, 0.4],
               dtype=np.float32).reshape([1, 2, 2, 2])
    )
    mean_data = np.array([10., 20.], dtype=np.float32).reshape([2])
    variance_data = np.array([0.25, 0.5], dtype=np.float32).reshape([2])
    beta_data = np.array([0.1, 0.6], dtype=np.float32).reshape([2])
    gamma_data = np.array([1., 2.], dtype=np.float32).reshape([2])

    # Create the non-deprecated part of the graph
    # (input, weights) --> Conv2D --> [...], plus inputs to [...]
    tf_g = tf.Graph()
    with tf_g.as_default():
      in_t = tf.constant(input_data, name="input_op")
      weights_t = tf.constant(weights_data, name="weights_op")
      conv_t = tf.nn.conv2d(in_t, weights_t, [1, 1, 1, 1], "VALID",
                            name="conv_op")
      _ = tf.constant(mean_data, name="mean_op")
      _ = tf.constant(variance_data, name="variance_op")
      _ = tf.constant(beta_data, name="beta_op")
      _ = tf.constant(gamma_data, name="gamma_op")
    g = gde.Graph(tf_g)

    # Add a deprecated batch normalization operator directly to the GraphDef,
    # since the Python APIs for making this op no longer exist.
    batch_norm_node = g.add_node("output", "BatchNormWithGlobalNormalization")
    batch_norm_node.set_inputs([g["conv_op:0"], g["mean_op:0"],
                                g["variance_op:0"], g["beta_op:0"],
                                g["gamma_op:0"]])
    batch_norm_node.add_attr("T", tf.float32)
    batch_norm_node.add_attr("variance_epsilon", 0.00001)
    batch_norm_node.add_attr("scale_after_normalization", False)

    # Run the graph before and after the rewrite and compare results
    with tf.Session(graph=g.to_tf_graph()) as sess:
      original_outputs = sess.run("output:0")
    gde.rewrite.fold_old_batch_norms(g)
    with tf.Session(graph=g.to_tf_graph()) as sess:
      fused_outputs = sess.run("output:0")
    self.assertClose(original_outputs, fused_outputs, delta=1e-5)

    # Make sure the rewrite happened
    for n in g.nodes:
      self.assertNotEqual(n.op_type, "BatchNormWithGlobalNormalization")


