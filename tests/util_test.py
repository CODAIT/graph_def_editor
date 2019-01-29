# Copyright 2018 IBM. All Rights Reserved.
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
"""Tests for tensorflow.contrib.graph_editor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import unittest

import graph_def_editor as gde


class UtilTest(unittest.TestCase):

  def test_list_view(self):
    """Test for gde.util.ListView."""
    l = [0, 1, 2]
    lv = gde.util.ListView(l)
    # Should not be the same id.
    self.assertIsNot(l, lv)
    # Should behave the same way than the original list.
    self.assertTrue(len(lv) == 3 and lv[0] == 0 and lv[1] == 1 and lv[2] == 2)
    # Should be read only.
    with self.assertRaises(TypeError):
      lv[0] = 0

  def test_is_iterable(self):
    """Test for gde.util.is_iterable."""
    self.assertTrue(gde.util.is_iterable([0, 1, 2]))
    self.assertFalse(gde.util.is_iterable(3))

  def test_unique_graph(self):
    """Test for gde.util.check_graphs and gde.util.get_unique_graph."""
    g0_graph = tf.Graph()
    with g0_graph.as_default():
      tf.constant(1, name="a")
      tf.constant(2, name="b")
    g1_graph = tf.Graph()
    with g1_graph.as_default():
      tf.constant(1, name="a")
      tf.constant(2, name="b")

    g0 = gde.Graph(g0_graph.as_graph_def())
    g1 = gde.Graph(g1_graph.as_graph_def())
    a0, b0, a1, b1 = (g0["a"], g0["b"], g1["a"], g1["b"])

    print("g0['a'] returns {} (type {})".format(g0['a'], type(g0['a'])))
    
    # Same graph, should be fine.
    self.assertIsNone(gde.util.check_graphs(a0, b0))
    # Two different graphs, should assert.
    with self.assertRaises(ValueError):
      gde.util.check_graphs(a0, b0, a1, b1)
    # a0 and b0 belongs to the same graph, should be fine.
    self.assertEqual(gde.util.get_unique_graph([a0, b0]), g0)
    # Different graph, should raise an error.
    with self.assertRaises(ValueError):
      gde.util.get_unique_graph([a0, b0, a1, b1])

  def test_make_list_of_node(self):
    """Test for gde.util.make_list_of_op."""
    g0_graph = tf.Graph()
    with g0_graph.as_default():
      tf.constant(1, name="a0")
      tf.constant(2, name="b0")
    g0 = gde.Graph(g0_graph)

    # Should extract the ops from the graph.
    self.assertEqual(len(gde.util.make_list_of_op(g0)), 2)
    # Should extract the ops from the tuple.
    self.assertEqual(len(gde.util.make_list_of_op((g0["a0"], g0["b0"]))), 2)

  def test_make_list_of_t(self):
    """Test for gde.util.make_list_of_t."""
    g0_graph = tf.Graph()
    with g0_graph.as_default():
      a0_op = tf.constant(1, name="a0")
      b0_op = tf.constant(2, name="b0")
      tf.add(a0_op, b0_op)
    g0 = gde.Graph(g0_graph)
    a0 = g0["a0"].output(0)
    b0 = g0["b0"].output(0)

    # Should extract the tensors from the graph.
    self.assertEqual(len(gde.util.make_list_of_t(g0)), 3)
    # Should extract the tensors from the tuple
    self.assertEqual(len(gde.util.make_list_of_t((a0, b0))), 2)
    # Should extract the tensors and ignore the ops.
    self.assertEqual(
        len(gde.util.make_list_of_t(
            (a0, a0.node, b0), ignore_ops=True)), 2)

  def test_get_generating_consuming(self):
    """Test for gde.util.get_generating_ops and gde.util.get_generating_ops."""
    g0_graph = tf.Graph()
    with g0_graph.as_default():
      a0_tensor = tf.constant(1, name="a0")
      b0_tensor = tf.constant(2, name="b0")
      tf.add(a0_tensor, b0_tensor, name="c0")
    g0 = gde.Graph(g0_graph)
    a0 = g0["a0"].output(0)
    b0 = g0["b0"].output(0)
    c0 = g0["c0"].output(0)

    self.assertEqual(len(gde.util.get_generating_ops([a0, b0])), 2)
    self.assertEqual(len(gde.util.get_consuming_ops([a0, b0])), 1)
    self.assertEqual(len(gde.util.get_generating_ops([c0])), 1)
    self.assertEqual(gde.util.get_consuming_ops([c0]), [])

  def test_control_outputs(self):
    """Test for the gde.util.ControlOutputs class."""
    g0_graph = tf.Graph()
    with g0_graph.as_default():
      a0_tensor = tf.constant(1, name="a0")
      b0_tensor = tf.constant(2, name="b0")
      x0_tensor = tf.constant(3, name="x0")
      with tf.control_dependencies([x0_tensor.op]):
        tf.add(a0_tensor, b0_tensor, name="c0")

    g0 = gde.Graph(g0_graph)
    x0_node = g0["x0"]
    c0_node = g0["c0"]
    control_outputs = gde.util.ControlOutputs(g0).get_all()
    self.assertEqual(len(control_outputs), 1)
    self.assertEqual(len(control_outputs[x0_node]), 1)
    self.assertIs(list(control_outputs[x0_node])[0], c0_node)

  def test_scope(self):
    """Test simple path scope functionalities."""
    self.assertEqual(gde.util.scope_finalize("foo/bar"), "foo/bar/")
    self.assertEqual(gde.util.scope_dirname("foo/bar/op"), "foo/bar/")
    self.assertEqual(gde.util.scope_basename("foo/bar/op"), "op")

  def test_placeholder(self):
    """Test placeholder functionalities."""
    g0_graph = tf.Graph()
    with g0_graph.as_default():
      tf.constant(1, name="foo")

    g0 = gde.Graph(g0_graph)
    a0 = g0["foo"].output(0)

    # Test placeholder name.
    self.assertEqual(gde.util.placeholder_name(a0), "geph__foo_0")
    self.assertEqual(gde.util.placeholder_name(None), "geph")
    self.assertEqual(
        gde.util.placeholder_name(
            a0, scope="foo/"), "foo/geph__foo_0")
    self.assertEqual(
        gde.util.placeholder_name(
            a0, scope="foo"), "foo/geph__foo_0")
    self.assertEqual(gde.util.placeholder_name(None, scope="foo/"), "foo/geph")
    self.assertEqual(gde.util.placeholder_name(None, scope="foo"), "foo/geph")

    # Test placeholder creation.
    g1_graph = tf.Graph()
    with g1_graph.as_default():
      tf.constant(1, dtype=tf.float32, name="a1")

    g1 = gde.Graph(g1_graph)
    a1_tensor = g1["a1"].output(0)
    print("Type of a1_tensor is {}".format(type(a1_tensor)))

    ph1 = gde.util.make_placeholder_from_tensor(g1, a1_tensor)
    ph2 = gde.util.make_placeholder_from_dtype_and_shape(g1, dtype=tf.float32)
    self.assertEqual(ph1.name, "geph__a1_0")
    self.assertEqual(ph2.name, "geph")

  def test_identity(self):
    tf_g = tf.Graph()
    with tf_g.as_default():
      c = tf.constant(42)
      i1 = tf.identity(c, name="identity_tf")

    g = gde.Graph(tf_g)
    i2_node = gde.util.make_identity(g, "identity_gde", g.get_tensor_by_name(c.name))
    i2 = i2_node.outputs[0]

    with g.to_tf_graph().as_default():
      with tf.Session() as sess:
        result1 = sess.run(i1.name)
        result2 = sess.run(i2.name)
    self.assertEqual(result1, result2)


if __name__ == "__main__":
  unittest.TestCase.main()
