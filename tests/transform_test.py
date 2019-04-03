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

import collections
import functools
import numpy as np

import tensorflow as tf
import unittest

import graph_def_editor as gde

# Precision tolerance for floating-point value tests.
ERROR_TOLERANCE = 1e-3


class TransformTest(unittest.TestCase):

  # Slightly modified version of the method by the same name in tf.TestCase
  def assertNear(self, f1, f2, err, msg=None):
    """Asserts that two floats are near each other.
    Checks that |f1 - f2| < err and asserts a test failure
    if not.
    Args:
      f1: A float value.
      f2: A float value.
      err: A float value.
      msg: An optional string message to append to the failure message.
    """
    # f1 == f2 is needed here as we might have: f1, f2 = inf, inf
    self.assertTrue(
      f1 == f2 or abs(f1 - f2) <= err,
      "{:f} != {:f} +/- {:f}{}".format(f1, f2, err,
                                       " ({})".format(msg if msg is not None
                                                      else "")))

  def setUp(self):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
      c0 = tf.constant(1.0, shape=[10], name="Const")
      c0.op._set_attr("_foo", tf.AttrValue(s=b"foo"))
      c1 = tf.constant(1.0, shape=[10], name="Const")
      c2 = tf.constant(1.0, shape=[10], name="Const")
      i = tf.constant(1.0, shape=[10], name="Input")
      tf.identity(tf.add(c2, tf.add(c1, tf.add(c0, i))), name="o")
    self.graph = gde.Graph(tf_graph)
    self.o = self.graph["o"]

  def test_copy(self):
    graph = gde.Graph()
    _, info = gde.copy(self.graph, graph)
    self.assertEqual(
        set(op.name for op in self.graph.nodes),
        set(op.name for op in graph.nodes))
    src_ops = self.graph.nodes
    dst_ops = graph.nodes
    for op in src_ops:
      op_ = info.transformed(op)
      self.assertTrue(op_ in dst_ops)
      self.assertEqual(op.name, op_.name)
      self.assertEqual(info.original(op_), op)
    src_ts = self.graph.tensors
    dst_ts = graph.tensors
    for t in src_ts:
      t_ = info.transformed(t)
      self.assertTrue(t_ in dst_ts)
      self.assertEqual(t.name, t_.name)
      self.assertEqual(info.original(t_), t)

  def test_copy_assert(self):
    tf_g = tf.Graph()
    with tf_g.as_default():
      a = tf.constant(1, name="a")
      b = tf.constant(1, name="b")
      eq = tf.equal(a, b, name="EQ")
      assert_tf_op = tf.Assert(eq, [a, b])
      with tf.control_dependencies([assert_tf_op]):
        _ = tf.add(a, b)
    assert_op_name = assert_tf_op.name

    g = gde.Graph(tf_g)
    assert_op = g[assert_op_name]
    sgv = gde.make_view([assert_op, g["EQ"], g["a"], g["b"]])
    copier = gde.Transformer()
    _, info = copier(sgv, sgv.graph, "", "")
    new_assert_op = info.transformed(assert_op)
    self.assertIsNotNone(new_assert_op)

  def test_transform(self):
    transformer = gde.Transformer()

    def my_transform_op_handler(info, op, new_inputs):
      add_noise = op.name.startswith("Add")
      op_, op_outputs_ = gde.transform.copy_op_handler(info, op, new_inputs)
      if not add_noise:
        return op_, op_outputs_

      # add some noise to op
      # Old code:
      # with info.graph_.as_default():
      #   t_ = math_ops.add(
      #       constant_op.constant(1.0, shape=[10], name="Noise"),
      #       op_.outputs[0],
      #       name="AddNoise")
      noise_op = gde.make_const(info.graph_, "Noise",
                                             np.full([10], 1., dtype=np.float32),
                                             uniquify_name=True)
      add_noise_op = info.graph_.add_node("AddNoise", "Add", uniquify_name=True)
      add_noise_op.add_attr("T", tf.float32)
      add_noise_op.set_inputs([noise_op.outputs[0], op_.outputs[0]])
      add_noise_op.infer_outputs()
      t_ = add_noise_op.outputs[0]

      # return the "noisy" op
      return op_, [t_]

    transformer.transform_op_handler = my_transform_op_handler

    graph = gde.Graph()
    transformer(self.graph, graph, "", "")
    matcher0 = gde.OpMatcher("AddNoise").input_ops(
        "Noise", gde.OpMatcher("Add").input_ops("Const", "Input"))
    matcher1 = gde.OpMatcher("AddNoise_1").input_ops(
        "Noise_1", gde.OpMatcher("Add_1").input_ops("Const_1", matcher0))
    matcher2 = gde.OpMatcher("AddNoise_2").input_ops(
        "Noise_2", gde.OpMatcher("Add_2").input_ops("Const_2", matcher1))
    top = gde.select_ops("^AddNoise_2$", graph=graph)[0]
    self.assertTrue(matcher2(top))

  def test_transform_nodedef_fn(self):
    transformer = gde.Transformer()

    def nodedef_fn(node_def):
      if "_foo" in node_def.attr:
        del node_def.attr["_foo"]
      node_def.attr["_bar"].s = b"bar"
      return node_def

    my_copy_op_handler = functools.partial(
        gde.transform.copy_op_handler, nodedef_fn=nodedef_fn)
    transformer.transform_op_handler = my_copy_op_handler

    graph = gde.Graph()
    transformer(self.graph, graph, "", "")

    c0_before = self.graph["Const"]
    c0_after = graph["Const"]
    self.assertEqual(c0_before.get_attr("_foo"), "foo")
    with self.assertRaises(ValueError):
      c0_after.get_attr("_foo")

    all_ops = graph.nodes
    for op in all_ops:
      self.assertEqual(op.get_attr("_bar"), "bar")

  def test_copy_with_input_replacements(self):
    # Original code:
    # with self.graph.as_default():
    #   _ = tf.constant(10.0, shape=[10], name="Input")
    # New code adds node as a NodeDef
    ten_node = gde.make_const(self.graph, "Ten", np.full([10], 10.,
                              dtype=np.float32))

    ten_tensor = ten_node.output(0)
    sgv, _ = gde.copy_with_input_replacements(
      # self.o is an identity on top of a tree of add ops
      [self.o, self.o.inputs[0].node],
      # Drill down to second input to outer add()
      {self.o.inputs[0].node.inputs[1]: ten_tensor}
    )

    after_graph = tf.Graph()
    with after_graph.as_default():
      tf.import_graph_def(self.graph.to_graph_def(), name="")
      with tf.Session() as sess:
        val = sess.run(sgv.outputs[0].name)

    print("val is {}".format(val))
    self.assertNear(
      np.linalg.norm(val - np.array([11])), 0.0, ERROR_TOLERANCE)


  def test_copy_with_collection(self):
    """Test for issue #36"""
    tmp_graph = tf.Graph()
    with tmp_graph.as_default():
      c = tf.constant(42, name="FortyTwo")
      tmp_graph.add_to_collection("Answers", c)

    g = gde.Graph(tmp_graph)
    g2 = gde.Graph()
    gde.transform.copy(g, g2)
    self.assertTrue("Answers" in g2.get_all_collection_keys())


  @staticmethod
  def _create_replace_graph():
    """Subroutine of the next few tests. Creates the graph that all these
    tests use. Since the tests modify the graph, it needs to be recreated
    each time.

    Returns:
      (Graph object, c, target tensor to replace, new value, output tensor)"""
    tmp_graph = tf.Graph()
    with tmp_graph.as_default():
      a = tf.constant(1.0, name="a")
      b = tf.Variable(1.0, name="b")
      eps = tf.constant(0.001, name="eps")
      tf.identity(a + b + eps, name="c")
      tf.constant(2.0, name="a_new")
    ret = gde.Graph(tmp_graph)
    return ret, ret["a"].output(0), ret["a_new"].output(0), ret["c"].output(0)

  def test_graph_replace(self):
    g, a, a_new, c = self._create_replace_graph()
    c_new = gde.graph_replace(c, {a: a_new})

    with g.to_tf_graph().as_default():
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c_val, c_new_val = sess.run([c.name, c_new.name])

    self.assertNear(c_val, 2.001, ERROR_TOLERANCE)
    self.assertNear(c_new_val, 3.001, ERROR_TOLERANCE)

  def test_graph_replace_dict(self):
    g, a, a_new, c = self._create_replace_graph()
    c_new = gde.graph_replace({"c": c}, {a: a_new})
    self.assertTrue(isinstance(c_new, dict))

    with g.to_tf_graph().as_default():
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c_val, c_new_val = sess.run([c.name,
                                     {k: v.name for k, v in c_new.items()}])

    self.assertTrue(isinstance(c_new_val, dict))
    self.assertNear(c_val, 2.001, ERROR_TOLERANCE)
    self.assertNear(c_new_val["c"], 3.001, ERROR_TOLERANCE)

  def test_graph_replace_ordered_dict(self):
    g, a, a_new, c = self._create_replace_graph()
    c_new = gde.graph_replace(collections.OrderedDict({"c": c}), {a: a_new})
    self.assertTrue(isinstance(c_new, collections.OrderedDict))

  def test_graph_replace_named_tuple(self):
    g, a, a_new, c = self._create_replace_graph()
    one_tensor = collections.namedtuple("OneTensor", ["t"])
    c_new = gde.graph_replace(one_tensor(c), {a: a_new})
    self.assertTrue(isinstance(c_new, one_tensor))

  def test_graph_replace_missing(self):
    tmp_graph = tf.Graph()
    with tmp_graph.as_default():
      a_tensor = tf.constant(1.0, name="a")
      b_tensor = tf.constant(2.0, name="b")
      _ = tf.add(a_tensor, 2 * b_tensor, name="c")
      _ = tf.constant(2.0, name="d")
    g = gde.Graph(tmp_graph)
    res = gde.graph_replace([g["b"].output(0), g["c"].output(0)],
                                         {g["a"].output(0): g["d"].output(0)})
    self.assertEqual(res[0].name, "b:0")
    self.assertEqual(res[1].name, "c_1:0")

  def test_graph_replace_gradients(self):
    tmp_graph = tf.Graph()
    with tmp_graph.as_default():
      w_tensor = tf.Variable(0.0, name="w")
      y_tensor = tf.multiply(tf.multiply(w_tensor, w_tensor, name="mul1"),
                             w_tensor, name="mul2")
      grad_tensor = tf.gradients(y_tensor, w_tensor, name="gradient")[0]
      _ = tf.identity(grad_tensor, "grad")

    g = gde.Graph(tmp_graph)

    # Extract the operations.
    replacement_ts = {g["w/read"].output(0): g["grad"].output(0)}

    # Should not raise exception.
    res = gde.graph_replace(g["grad"].output(0), replacement_ts,
                                         dst_scope="res")

    self.assertNotEqual(res.name, g["grad"].output(0).name)
    after_graph = tf.Graph()
    with after_graph.as_default():
      tf.import_graph_def(g.to_graph_def(), name="")
      gde.util.load_variables_to_tf_graph(g)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        g_val, res_val = sess.run([g["grad"].output(0).name, res.name])
    self.assertNear(g_val, 0.0, ERROR_TOLERANCE)
    self.assertNear(res_val, 0.0, ERROR_TOLERANCE)

  def test_graph_while_loop(self):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
      max_index = tf.placeholder(dtype=tf.int32, shape=tuple())
      index_start = tf.constant(1)
      sum_start = tf.constant(0)
      _, result = tf.while_loop(
          cond=lambda i, unused_s: i <= max_index,
          body=lambda i, s: (i + 1, s + i),
          loop_vars=[index_start, sum_start])
    g = gde.Graph(tf_graph)
    result_tensor = g[result.op.name].output(0)
    max_index_tensor = g[max_index.op.name].output(0)

    g.frozen = True
    copied_graph = gde.Graph()
    _, copy_info = gde.copy(
        g, dst_graph=copied_graph, dst_scope="imported")
    copied_result_tensor = copy_info.transformed(result_tensor)
    copied_max_index_tensor = copy_info.transformed(max_index_tensor)

    tf_copied_graph = tf.Graph()
    with tf_copied_graph.as_default():
      tf.import_graph_def(copied_graph.to_graph_def(), name="")
      with tf.Session() as sess:
        n = 10
        sum_val = sess.run(copied_result_tensor.name,
                           feed_dict={copied_max_index_tensor.name: n})
        self.assertEqual(sum_val, 55)

  def test_graph_cond(self):
    tf_g = tf.Graph()
    with tf_g.as_default():
      choice_tensor = tf.placeholder(shape=(), dtype=tf.bool, name="choice")
      _ = tf.identity(
        tf.cond(
          choice_tensor,
          lambda: tf.constant(1),
          lambda: tf.constant(2)
        ),
        name="result"
      )

    g = gde.Graph(tf_g)
    choice = g["choice"].output(0)
    result = g["result"].output(0)

    copied_g = gde.Graph()
    _, copy_info = gde.copy(
        g, dst_graph=copied_g, dst_scope="imported")
    copied_result = copy_info.transformed(result)
    copied_choice = copy_info.transformed(choice)

    tf_copied_graph = tf.Graph()
    with tf_copied_graph.as_default():
      tf.import_graph_def(copied_g.to_graph_def(), name="")
      with tf.Session() as sess:
        res = sess.run(copied_result.name, feed_dict={copied_choice.name: True})
        self.assertEqual(res, 1)
        res = sess.run(copied_result.name,
                       feed_dict={copied_choice.name: False})
        self.assertEqual(res, 2)


if __name__ == "__main__":
  unittest.main()
