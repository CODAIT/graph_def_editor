# Copyright 2021 Google. All Rights Reserved.
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
"""Tests for function_graph.py in the GraphDef Editor."""

import unittest
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import shutil
import tempfile
import numpy as np

import graph_def_editor as gde


class FunctionGraphTest(unittest.TestCase):

  def setUp(self):
    # Create a temporary directory for SavedModel files.
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    # Remove the directory after the test.
    # Comment out this line to prevent deleting temps.
    shutil.rmtree(self.temp_dir)
    pass  # In case previous line gets commented out

  def build_tf_graph(self):
    """Builds a tf graph for function (x + y) * 10.0 ."""
    @tf.function
    def multiplier_function(x):
      return tf.constant(10.0, name="function_multiplier") * x

    tf_g = tf.Graph()
    with tf_g.as_default():
      x = tf.placeholder(name="x", dtype=tf.float32, shape=[])
      y = tf.placeholder(name="y", dtype=tf.float32, shape=[])
      result_op = tf.add(x, y, name="add")
      _ = multiplier_function(result_op)
    return tf_g

  def run_tf_graph(self, tf_g, x, y):
    with tf.Session(graph=tf_g) as sess:
      x_tensor = tf_g.get_tensor_by_name("x:0")
      y_tensor = tf_g.get_tensor_by_name("y:0")
      output_tensor = tf_g.get_tensor_by_name("PartitionedCall:0")
      return sess.run(output_tensor, {x_tensor: x, y_tensor: y})

  def save_tf_graph(self, tf_g, model_dir):
    x_tensor = tf_g.get_tensor_by_name("x:0")
    y_tensor = tf_g.get_tensor_by_name("y:0")
    output_tensor = tf_g.get_tensor_by_name("PartitionedCall:0")
    with tf.Session(graph=tf_g) as sess:
      tf.saved_model.simple_save(sess, model_dir,
                                 inputs={"x": x_tensor, "y": y_tensor},
                                 outputs={"out": output_tensor})

  def test_function_rewrite(self):
    tf_g = self.build_tf_graph()
    self.assertEqual(30.0, self.run_tf_graph(tf_g, 1.0, 2.0))
    graph = gde.Graph(tf_g)
    add_op = graph.get_node_by_name("add")
    function_name = add_op.outputs[0].consumers()[0].get_attr("f").name
    self.assertIn(function_name, graph.function_names)

    function_graph = graph.get_function_graph_by_name(function_name)
    function_multiplier_op = \
        function_graph.get_node_by_name("function_multiplier")
    self.assertEqual(10.0, function_multiplier_op.get_attr("value"))
    function_multiplier_op.replace_attr("value",
                                        np.array(1000.0, dtype=np.float32))

    self.assertEqual(3000.0, self.run_tf_graph(graph.to_tf_graph(), 1.0, 2.0))
    return graph

  def test_export_saved_model(self):
    g = self.test_function_rewrite()
    model_dir = self.temp_dir + "/saved_model"
    g.to_saved_model(model_dir)
    tf_g = tf.Graph()
    with tf.Session(graph=tf_g) as sess:
      _ = tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING],
                              model_dir)
    self.assertEqual(3000.0, self.run_tf_graph(tf_g, 1.0, 2.0))

  def test_import_saved_model(self):
    g = self.test_function_rewrite()
    model_dir = self.temp_dir + "/saved_model"
    self.save_tf_graph(g.to_tf_graph(), model_dir)

    g = gde.saved_model_to_graph(model_dir)
    self.assertEqual(3000.0, self.run_tf_graph(g.to_tf_graph(), 1.0, 2.0))

  def test_number_attr_support(self):
    model_dir = self.temp_dir + "/saved_model"

    @tf.function
    def test_function(c):
      cdim = tf.constant(1, tf.int32)
      c1 = tf.constant([2, 1, 5], tf.int32, name="FuncConst")
      c2 = tf.constant([2, 1, 5], tf.int32)
      # ConcatOffset has variable number of intputs and outputs
      # that is using number_attr in functions
      concat_offset = tf.raw_ops.ConcatOffset(
          concat_dim=cdim, shape=[c, c1, c2])
      out = tf.math.reduce_sum(concat_offset)
      return out

    tf_g = tf.Graph()
    with tf_g.as_default():
      with tf.Session() as sess:
        c = tf.placeholder(name="c", dtype=tf.int32)
        out_func = test_function(c)
        c = tf_g.get_tensor_by_name("c:0")
        self.assertEqual(3, sess.run(out_func, {c: [2, 1, 5]}))

        tf.saved_model.simple_save(
            sess, model_dir, inputs={"c": c}, outputs={"out_func": out_func})

    g = gde.saved_model_to_graph(model_dir)

    tf_g = g.to_tf_graph()
    with tf.Session(graph=tf_g) as sess:
      output_tensor = tf_g.get_tensor_by_name("PartitionedCall:0")
      c = tf_g.get_tensor_by_name("c:0")
      self.assertEqual(3, sess.run(output_tensor, {c: [2, 1, 5]}))

    f = g.get_function_graph_by_name(g.function_names[0])
    func_const_op = f.get_node_by_name("FuncConst")
    func_const_op.replace_attr("value", np.array([2, 2, 5], dtype=np.int32))

    tf_g = g.to_tf_graph()
    with tf.Session(graph=tf_g) as sess:
      output_tensor = tf_g.get_tensor_by_name("PartitionedCall:0")
      c = tf_g.get_tensor_by_name("c:0")
      self.assertEqual(4, sess.run(output_tensor, {c: [2, 1, 5]}))

if __name__ == "__main__":
  unittest.main()
