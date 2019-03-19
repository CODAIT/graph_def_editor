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

  def assertClose(self,
                  expected, # type: np.ndarray
                  actual, # type: np.ndarray
                  delta, # type: float
                  ):
    """
    Assert that all values in two arrays are within a certain distance of
    each other.
    """
    abs_diff = np.abs(expected - actual)
    max_diff_flat_ix = np.argmax(abs_diff)
    max_diff_ix = np.unravel_index(max_diff_flat_ix, abs_diff.shape)
    max_diff = abs_diff[max_diff_ix]
    if max_diff > delta:
      self.fail(msg="Maximum difference of {} at index {} is greater than "
                    "tolerance {} when comparing\n"
                    "{}\n"
                    "  and\n"
                    "{}".format(max_diff, max_diff_ix, delta, expected, actual))

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
    with tf.Session(graph=g.to_tf_graph()) as sess:
      fused_outputs = sess.run(output_t.name)

    self.assertClose(original_outputs, fused_outputs, delta=1e-5)

    # Make sure the rewrite happened
    for n in g.nodes:
      self.assertNotEqual(n.op_type, "Mul")

  def test_fold_batch_norms_conv_2d_shared(self):
    """
    Python port of TestFoldBatchNormsConv2DShared in the TF Graph Transform Tool
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
    mul_values_data_2 = (
      np.array([1., 2.], dtype=np.float32).reshape([2])
    )

    # Create and run graph:
    # (input, weights) --> Conv2D --> Mul(const)
    #                         |-----> Mul(const)
    tf_g = tf.Graph()
    with tf_g.as_default():
      in_t = tf.constant(input_data, name="input_op")
      weights_t = tf.constant(weights_data, name="weights_op")
      conv_t = tf.nn.conv2d(in_t, weights_t, [1, 1, 1, 1], "VALID",
                            name="conv_op")
      mul_values_t = tf.constant(mul_values_data, name="mul_values")
      output_t = tf.multiply(conv_t, mul_values_t, name="output")
      mul_values_2_t = tf.constant(mul_values_data_2, name="mul_values_2")
      output_2_t = tf.multiply(conv_t, mul_values_2_t, name="output_2")
    with tf.Session(graph=tf_g) as sess:
      original_outputs = sess.run([output_t, output_2_t])

    # Rewrite and compare results
    g = gde.Graph(tf_g)
    gde.rewrite.fold_batch_norms(g)
    with tf.Session(graph=g.to_tf_graph()) as sess:
      fused_outputs = sess.run([output_t.name, output_2_t.name])

    self.assertClose(original_outputs[0], fused_outputs[0], delta=1e-5)
    self.assertClose(original_outputs[1], fused_outputs[1], delta=1e-5)

  def test_fold_batch_norms_mat_mul(self):
    """
    Python port of TestFoldBatchNormsMatMul in the TF Graph Transform Tool
    tests.
    """
    input_data = (
      np.array([1., 4., 2., 5., 3., 6., -1., -4., -2., -5., -3., -6.],
               dtype=np.float32).reshape([6, 2])
    )
    weights_data = (
      np.array([1., 2., 3., 4.],
               dtype=np.float32).reshape([2, 2])
    )
    mul_values_data = (
      np.array([2., 3.], dtype=np.float32).reshape([2])
    )

    # Create and run graph:
    # (input, weights) --> MatMul --> Mul(const)
    tf_g = tf.Graph()
    with tf_g.as_default():
      in_t = tf.constant(input_data, name="input_op")
      weights_t = tf.constant(weights_data, name="weights_op")
      matmul_t = tf.linalg.matmul(in_t, weights_t, name="matmul_op")
      mul_values_t = tf.constant(mul_values_data, name="mul_values")
      output_t = tf.multiply(matmul_t, mul_values_t, name="output")
    with tf.Session(graph=tf_g) as sess:
      original_outputs = sess.run(output_t)

    # Rewrite and compare results
    g = gde.Graph(tf_g)
    gde.rewrite.fold_batch_norms(g)
    with tf.Session(graph=g.to_tf_graph()) as sess:
      fused_outputs = sess.run(output_t.name)

    self.assertClose(original_outputs, fused_outputs, delta=1e-5)

    # Make sure the rewrite happened
    for n in g.nodes:
      self.assertNotEqual(n.op_type, "Mul")

  def test_fold_old_batch_norms(self):
    """
    Python port of TestFoldOldBatchNorms() in the TF Graph Transform Tool
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
      mean_t = tf.constant(mean_data, name="mean_op")
      variance_t = tf.constant(variance_data, name="variance_op")
      beta_t = tf.constant(beta_data, name="beta_op")
      gamma_t = tf.constant(gamma_data, name="gamma_op")

    # Set the producer version of the graph to a version where the
    # BatchNormWithGlobalNormalization op still existed.
    graph_def = tf_g.as_graph_def()
    graph_def.versions.producer = 8
    g = gde.Graph(graph_def)

    # Add a deprecated batch normalization operator directly to the GraphDef,
    # since the Python APIs for making this op no longer exist.
    batch_norm_node = g.add_node("output", "BatchNormWithGlobalNormalization")
    batch_norm_node.set_inputs([g[conv_t.name], g[mean_t.name],
                                g[variance_t.name], g[beta_t.name],
                                g[gamma_t.name]])
    batch_norm_node.add_attr("T", tf.float32)
    batch_norm_node.add_attr("variance_epsilon", 0.00001)
    batch_norm_node.add_attr("scale_after_normalization", False)
    # Can't infer output shapes because this op no longer exists in the tf
    # Python API. So copy output info from the convolution op.
    conv_node = g.get_node_by_name(conv_t.op.name)
    batch_norm_node.set_outputs_from_pairs(
      [(t.dtype, t.shape) for t in conv_node.outputs])

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

  def test_fold_fused_batch_norms(self):
    """
    Version of test_fold_old_batch_norms() with a FusedBatchNorms op instead
    of BatchNormWithGlobalNormalization
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
      mean_t = tf.constant(mean_data, name="mean_op")
      variance_t = tf.constant(variance_data, name="variance_op")
      beta_t = tf.constant(beta_data, name="beta_op")
      gamma_t = tf.constant(gamma_data, name="gamma_op")
    g = gde.Graph(tf_g)

    # Add fused batch norm node manually because there's no Python API to add
    # this op directly.
    batch_norm_node = g.add_node("output", "FusedBatchNorm")
    batch_norm_node.set_inputs([g[conv_t.name], g[gamma_t.name],
                                g[beta_t.name], g[mean_t.name],
                                g[variance_t.name]])
    batch_norm_node.add_attr("T", tf.float32)
    batch_norm_node.add_attr("epsilon", 0.00001)
    batch_norm_node.add_attr("is_training", False)
    batch_norm_node.infer_outputs()

    # Run the graph before and after the rewrite and compare results
    with tf.Session(graph=g.to_tf_graph()) as sess:
      original_outputs = sess.run("output:0")
    gde.rewrite.fold_old_batch_norms(g)
    with tf.Session(graph=g.to_tf_graph()) as sess:
      fused_outputs = sess.run("output:0")
    self.assertClose(original_outputs, fused_outputs, delta=1e-5)

    # Make sure the rewrite happened
    for n in g.nodes:
      self.assertNotEqual(n.op_type, "FusedBatchNorm")

  def test_fold_fused_batch_norms_depthwise(self):
    """
    Version of test_fold_fused_batch_norms() with a depthwise convolution op
    """
    # We try this test with channel multipliers of 1 and 2
    def run_test(channel_multiplier_is_one # type: bool
                 ):
      input_data = (
        np.array([1., 4., 2., 5., 3., 6., -1., -4., -2., -5., -3., -6.],
                 dtype=np.float32).reshape([1, 1, 6, 2])
      )
      if channel_multiplier_is_one:
        weights_data = (np.array([1., 2., 3., 4.],
                                 dtype=np.float32).reshape([1, 2, 2, 1]))
        mean_data = np.array([10., 20.], dtype=np.float32).reshape([2])
        variance_data = np.array([0.25, 0.5], dtype=np.float32).reshape([2])
        beta_data = np.array([0.1, 0.6], dtype=np.float32).reshape([2])
        gamma_data = np.array([1., 2.], dtype=np.float32).reshape([2])
      else:
        weights_data = (np.array([1., 2., 3., 4., 0.1, 0.2, 0.3, 0.4],
                                 dtype=np.float32).reshape([1, 2, 2, 2]))
        mean_data = np.array([10., 20., 30., 40.],
                             dtype=np.float32).reshape([4])
        variance_data = np.array([0.25, 0.5, 1.0, 1.5],
                                 dtype=np.float32).reshape([4])
        beta_data = np.array([0.1, 0.2, 0.3, 0.6],
                             dtype=np.float32).reshape([4])
        gamma_data = np.array([1., 2., 3., 4.], dtype=np.float32).reshape([4])

      # Create the non-deprecated part of the graph
      # (input, weights) --> Conv2D --> [...], plus inputs to [...]
      tf_g = tf.Graph()
      with tf_g.as_default():
        in_t = tf.constant(input_data, name="input_op")
        weights_t = tf.constant(weights_data, name="weights_op")
        conv_t = tf.nn.depthwise_conv2d(in_t, weights_t, [1, 1, 1, 1],
                                        "VALID", name="conv_op")
        mean_t = tf.constant(mean_data, name="mean_op")
        variance_t = tf.constant(variance_data, name="variance_op")
        beta_t = tf.constant(beta_data, name="beta_op")
        gamma_t = tf.constant(gamma_data, name="gamma_op")
      g = gde.Graph(tf_g)

      # Add fused batch norm node manually because there's no Python API to add
      # this op directly.
      batch_norm_node = g.add_node("output", "FusedBatchNorm")
      batch_norm_node.set_inputs([g[conv_t.name], g[gamma_t.name],
                                  g[beta_t.name], g[mean_t.name],
                                  g[variance_t.name]])
      batch_norm_node.add_attr("T", tf.float32)
      batch_norm_node.add_attr("epsilon", 0.00001)
      batch_norm_node.add_attr("is_training", False)
      batch_norm_node.infer_outputs()

      # Run the graph before and after the rewrite and compare results
      with tf.Session(graph=g.to_tf_graph()) as sess:
        original_outputs = sess.run("output:0")
      gde.rewrite.fold_old_batch_norms(g)
      with tf.Session(graph=g.to_tf_graph()) as sess:
        fused_outputs = sess.run("output:0")
      self.assertClose(original_outputs, fused_outputs,
                       delta=(2e-5 if channel_multiplier_is_one else 4e-5))

      # Make sure the rewrite happened.
      for n in g.nodes:
        self.assertNotEqual(n.op_type, "FusedBatchNorm")

    run_test(False)
    run_test(True)

  def test_fold_old_batch_norms_with_concat(self):
    """
    Python port of TestFoldFusedBatchNormsWithConcat() in the TF Graph
    Transform Tool tests.
    """
    # Run the test twice, changing how we concatenate the outputs of our two
    # convolutions.
    def run_test(split # type: bool
                 ):
      """
      Args:
        split: if True, concatenate along channels dimension
      """
      input_shape = [1, 1, 6, 2] if split else [1, 1, 12, 1]
      input_data = (
        np.array([1., 4., 2., 5., 3., 6., -1., -4., -2., -5., -3., -6.],
                 dtype=np.float32).reshape(input_shape)
      )
      weights_shape = [1, 2, 2, 1] if split else [1, 2, 1, 2]
      weights_data = (
        np.array([1., 2., 3., 4.], dtype=np.float32).reshape(weights_shape)
      )
      mean_data = np.array([10., 20.], dtype=np.float32).reshape([2])
      variance_data = np.array([0.25, 0.5], dtype=np.float32).reshape([2])
      beta_data = np.array([0.1, 0.6], dtype=np.float32).reshape([2])
      gamma_data = np.array([1., 2.], dtype=np.float32).reshape([2])
      concat_axis = 3 if split else 2

      # Create the parts below the batch norm using TF APIs
      # 2 X ((input, weights) --> Conv2D) --> Concat -> [...],
      # plus inputs to [...]
      tf_g = tf.Graph()
      with tf_g.as_default():
        in0_t = tf.constant(input_data, name="input_0_op")
        weights0_t = tf.constant(weights_data, name="weights_0_op")
        conv0_t = tf.nn.conv2d(in0_t, weights0_t, [1, 1, 1, 1], "VALID",
                              name="conv_0_op")
        in1_t = tf.constant(input_data, name="input_1_op")
        weights1_t = tf.constant(weights_data, name="weights_1_op")
        conv1_t = tf.nn.conv2d(in1_t, weights1_t, [1, 1, 1, 1], "VALID",
                               name="conv_1_op")
        concat_t = tf.concat([conv0_t, conv1_t], concat_axis, name="concat_op")
        mean_t = tf.constant(mean_data, name="mean_op")
        variance_t = tf.constant(variance_data, name="variance_op")
        beta_t = tf.constant(beta_data, name="beta_op")
        gamma_t = tf.constant(gamma_data, name="gamma_op")

      g = gde.Graph(tf_g)

      # Now add the FusedBatchNorm node directly, since there's no TF API to
      # create that op.
      batch_norm_node = g.add_node("output", "FusedBatchNorm")
      batch_norm_node.set_inputs([g[concat_t.name], g[gamma_t.name],
                                  g[beta_t.name], g[mean_t.name],
                                  g[variance_t.name]])
      batch_norm_node.add_attr("T", tf.float32)
      batch_norm_node.add_attr("epsilon", 0.00001)
      batch_norm_node.add_attr("is_training", False)
      batch_norm_node.infer_outputs()

      # Run the graph before and after the rewrite and compare results
      with tf.Session(graph=g.to_tf_graph()) as sess:
        original_outputs = sess.run("output:0")
      gde.rewrite.fold_old_batch_norms(g)
      with tf.Session(graph=g.to_tf_graph()) as sess:
        fused_outputs = sess.run("output:0")
      self.assertClose(original_outputs, fused_outputs, delta=2e-5)

      # Make sure the rewrite happened
      for n in g.nodes:
        self.assertNotEqual(n.op_type, "FusedBatchNorm")

    # Invoke the function that we just defined twice.
    run_test(split=True)
    run_test(split=False)

  def test_fold_old_batch_norms_with_batch_to_space(self):
    """
    Python port of TestFoldFusedBatchNormsWithBatchToSpace() in the TF Graph
    Transform Tool tests.
    """
    input_data = (
      np.array([1., 4., 2., 5., 3., 6., -1., -4., -2., -5., -3., -6.],
               dtype=np.float32).reshape([2, 1, 3, 2])
    )
    weights_data = (
      np.array([1., 2., 3., 4., 0.1, 0.2, 0.3, 0.4],
               dtype=np.float32).reshape([1, 2, 2, 2])
    )
    mean_data = np.array([10., 20.], dtype=np.float32).reshape([2])
    variance_data = np.array([0.25, 0.5], dtype=np.float32).reshape([2])
    beta_data = np.array([0.1, 0.6], dtype=np.float32).reshape([2])
    gamma_data = np.array([1., 2.], dtype=np.float32).reshape([2])
    block_shape_data = np.array([1, 2]).reshape([2])
    crops_data = np.array([0, 0, 0, 1]).reshape([2, 2])

    # Create the parts below the batch norm using TF APIs
    # (input, weights) --> Conv2D --> BatchToSpaceND --> [...],
    # plus inputs to [...]
    tf_g = tf.Graph()
    with tf_g.as_default():
      in_t = tf.constant(input_data, name="input_op")
      weights_t = tf.constant(weights_data, name="weights_op")
      conv_t = tf.nn.conv2d(in_t, weights_t, [1, 1, 1, 1], "VALID",
                            name="conv_op")
      batch_to_space_t = tf.batch_to_space_nd(conv_t, block_shape_data,
                                              crops_data,
                                              name="batch_to_space_op")
      mean_t = tf.constant(mean_data, name="mean_op")
      variance_t = tf.constant(variance_data, name="variance_op")
      beta_t = tf.constant(beta_data, name="beta_op")
      gamma_t = tf.constant(gamma_data, name="gamma_op")

    g = gde.Graph(tf_g)

    # Now add the FusedBatchNorm node directly, since there's no TF API to
    # create that op.
    batch_norm_node = g.add_node("output", "FusedBatchNorm")
    batch_norm_node.set_inputs([g[batch_to_space_t.name], g[gamma_t.name],
                                g[beta_t.name], g[mean_t.name],
                                g[variance_t.name]])
    batch_norm_node.add_attr("T", tf.float32)
    batch_norm_node.add_attr("epsilon", 0.00001)
    batch_norm_node.add_attr("is_training", False)
    batch_norm_node.infer_outputs()

    # Run the graph before and after the rewrite and compare results
    with tf.Session(graph=g.to_tf_graph()) as sess:
      original_outputs = sess.run("output:0")
    gde.rewrite.fold_old_batch_norms(g)
    with tf.Session(graph=g.to_tf_graph()) as sess:
      fused_outputs = sess.run("output:0")
    self.assertClose(original_outputs, fused_outputs, delta=1e-5)

    # Make sure the rewrite happened
    for n in g.nodes:
      self.assertNotEqual(n.op_type, "FusedBatchNorm")

  def test_fold_batch_norms_up(self):
    """
    Test of the fold_batch_norms_up() rewrite with the pattern:
      Mul => Add => Conv2D
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
    add_values_data = (
      np.array([0.1, 0.2], dtype=np.float32).reshape([2])
    )

    # Create and run graph:
    # input -> Mul -> Add -> Conv2D(weights)
    tf_g = tf.Graph()
    with tf_g.as_default():
      in_t = tf.constant(input_data, name="input_op")
      mul_t = tf.multiply(in_t, mul_values_data, name="mul_op")
      add_t = tf.add(mul_t, add_values_data, name="add_op")
      weights_t = tf.constant(weights_data, name="weights_op")
      output_t = tf.nn.conv2d(add_t, weights_t, [1, 1, 1, 1], "VALID",
                              name="output")
    with tf.Session(graph=tf_g) as sess:
      original_outputs = sess.run(output_t)

    # Rewrite and compare results
    g = gde.Graph(tf_g)
    gde.rewrite.fold_batch_norms_up(g)
    with tf.Session(graph=g.to_tf_graph()) as sess:
      fused_outputs = sess.run(output_t.name)

    self.assertClose(original_outputs, fused_outputs, delta=1e-5)

    # Make sure the rewrite happened
    for n in g.nodes:
      self.assertNotEqual(n.op_type, "Mul")

  def test_fold_batch_norms_up_fused(self):
    """
    Test of the fold_batch_norms_up() rewrite with the pattern:
       FusedBatchNorm => Relu => Conv2D
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
    # input -> Relu -> Conv2D
    tf_g = tf.Graph()
    with tf_g.as_default():
      in_t = tf.constant(input_data, name="input_op")
      relu_t = tf.nn.relu(in_t, name="relu_op")
      weights_t = tf.constant(weights_data, name="weights_op")
      conv_t = tf.nn.conv2d(relu_t, weights_t, [1, 1, 1, 1], "VALID",
                            name="output")
      mean_t = tf.constant(mean_data, name="mean_op")
      variance_t = tf.constant(variance_data, name="variance_op")
      beta_t = tf.constant(beta_data, name="beta_op")
      gamma_t = tf.constant(gamma_data, name="gamma_op")
    g = gde.Graph(tf_g)

    # Add fused batch norm node manually because there's no Python API to add
    # this op directly.
    batch_norm_node = g.add_node("batch_norm_op", "FusedBatchNorm")
    batch_norm_node.set_inputs([g[in_t.name], g[gamma_t.name],
                                g[beta_t.name], g[mean_t.name],
                                g[variance_t.name]])
    batch_norm_node.add_attr("T", tf.float32)
    batch_norm_node.add_attr("epsilon", 0.00001)
    batch_norm_node.add_attr("is_training", False)
    batch_norm_node.infer_outputs()

    # Redirect the input of the ReLU to our new batch norm
    g.get_node_by_name(relu_t.op.name).set_inputs([batch_norm_node.output(0)])

    # Run the graph before and after the rewrite and compare results
    with tf.Session(graph=g.to_tf_graph()) as sess:
      original_outputs = sess.run("output:0")
    gde.rewrite.fold_batch_norms_up(g)
    with tf.Session(graph=g.to_tf_graph()) as sess:
      fused_outputs = sess.run("output:0")
    self.assertClose(original_outputs, fused_outputs, delta=1e-5)

    # Make sure the rewrite happened
    for n in g.nodes:
      self.assertNotEqual(n.op_type, "FusedBatchNorm")

  def test_fold_batch_norms_up_fused_relu6(self):
    """
    Test of the fold_batch_norms_up() rewrite with the pattern:
       FusedBatchNorm => Relu6 => Conv2D
    """
    # Note that 3 and 5 changed to 30 and 50 to trigger Relu6
    input_data = (
      np.array([1., 4., 10., 50., 30., 6., -1., -4., -2., -5., -3., -6.],
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
    # input -> Relu -> DepthwiseConv2D
    tf_g = tf.Graph()
    with tf_g.as_default():
      in_t = tf.constant(input_data, name="input_op")
      relu6_t = tf.nn.relu6(in_t, name="relu6_op")
      weights_t = tf.constant(weights_data, name="weights_op")
      _ = tf.nn.depthwise_conv2d(relu6_t, weights_t, [1, 1, 1, 1], "VALID",
                                 name="output")
      mean_t = tf.constant(mean_data, name="mean_op")
      variance_t = tf.constant(variance_data, name="variance_op")
      beta_t = tf.constant(beta_data, name="beta_op")
      gamma_t = tf.constant(gamma_data, name="gamma_op")
    g = gde.Graph(tf_g)

    # Add fused batch norm node manually because there's no Python API to add
    # this op directly.
    batch_norm_node = g.add_node("batch_norm_op", "FusedBatchNorm")
    batch_norm_node.set_inputs([g[in_t.name], g[gamma_t.name],
                                g[beta_t.name], g[mean_t.name],
                                g[variance_t.name]])
    batch_norm_node.add_attr("T", tf.float32)
    batch_norm_node.add_attr("epsilon", 0.00001)
    batch_norm_node.add_attr("is_training", False)
    batch_norm_node.infer_outputs()

    # Redirect the input of the ReLU to our new batch norm
    g.get_node_by_name(relu6_t.op.name).set_inputs([batch_norm_node.output(0)])

    # Run the graph before and after the rewrite and compare results
    with tf.Session(graph=g.to_tf_graph()) as sess:
      original_outputs = sess.run("output:0")
      relu6_inputs = sess.run("batch_norm_op:0")
    gde.rewrite.fold_batch_norms_up(g)
    with tf.Session(graph=g.to_tf_graph()) as sess:
      fused_outputs = sess.run("output:0")
    self.assertClose(original_outputs, fused_outputs, delta=1e-5)

    # Make sure input to Relu6 op was large enough to trigger the "6" part.
    self.assertTrue(np.any(relu6_inputs > 6.))

    # Make sure the rewrite happened
    for n in g.nodes:
      self.assertNotEqual(n.op_type, "FusedBatchNorm")
