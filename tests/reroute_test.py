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

import numpy as np
import tensorflow as tf
import unittest

import graph_def_editor as gde


class RerouteTest(unittest.TestCase):

  def setUp(self):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
      a0 = tf.constant(1.0, shape=[2], name="a0")
      b0 = tf.constant(2.0, shape=[2], name="b0")
      _ = tf.add(a0, b0, name="c0")
      a1 = tf.constant(3.0, shape=[2], name="a1")
      b1 = tf.constant(4.0, shape=[2], name="b1")
      _ = tf.add(a1, b1, name="c1")
      a2 = tf.constant(3.0, shape=[3], name="a2")
      b2 = tf.constant(4.0, shape=[3], name="b2")
      _ = tf.add(a2, b2, name="c2")

    self.graph = gde.Graph(tf_graph)
    # Programmatically add all the tensors as fields of this object.
    for letter in ["a", "b", "c"]:
      for number in ["0", "1", "2"]:
        op_name = letter + number
        self.__dict__[op_name] = self.graph[op_name].output(0)

  def test_swap(self):
    gde.swap_ts([self.a0, self.b0], [self.a1, self.b1])
    self.assertTrue(gde.OpMatcher("c0").input_ops("a1", "b1")(self.c0.op))
    self.assertTrue(gde.OpMatcher("c1").input_ops("a0", "b0")(self.c1.op))

  def test_multiswap(self):
    # Original code:
    # with self.graph.as_default():
    #   a3 = constant_op.constant(3.0, shape=[2], name="a3")
    # New code adds a NodeDef to the graph:
    a3_node = gde.make_const(self.graph, "a3", np.full([2], 3.0,
                                                                    dtype=np.float32))

    gde.swap_ios(gde.sgv(a3_node).remap_outputs([0, 0]),
                              gde.sgv(self.a0.op, self.a1.op))
    self.assertTrue(gde.OpMatcher("c0").input_ops("a3", "b0")(self.c0.op))
    self.assertTrue(gde.OpMatcher("c1").input_ops("a3", "b1")(self.c1.op))

  def test_reroute(self):
    gde.reroute_ts([self.a0, self.b0], [self.a1, self.b1])
    self.assertTrue(gde.OpMatcher("c0").input_ops("a0", "b0")(self.c0.op))
    self.assertTrue(gde.OpMatcher("c1").input_ops("a0", "b0")(self.c1.op))

    gde.reroute_ts([self.a1, self.b1], [self.a0, self.b0])
    self.assertTrue(gde.OpMatcher("c0").input_ops("a1", "b1")(self.c0.op))
    self.assertTrue(gde.OpMatcher("c1").input_ops("a1", "b1")(self.c1.op))

  def test_compatibility(self):
    with self.assertRaises(ValueError):
      gde.reroute_ts([self.a0, self.b0], [self.a2, self.b2])

  def test_reroute_can_modify(self):
    # create a special graph where "a" is an ambiguous tensor. That is
    # it is both an input and an output of the ops in sgv0.
    tf_graph = tf.Graph()
    with tf_graph.as_default():
      a_tensor = tf.constant(1.0, shape=[2], name="a")
      b_tensor = tf.constant(2.0, shape=[2], name="b")
      c_tensor = tf.add(a_tensor, b_tensor, name="c")
      _ = tf.add(a_tensor, c_tensor, name="d")
      e_tensor = tf.constant(1.0, shape=[2], name="e")
      f_tensor = tf.constant(2.0, shape=[2], name="f")
      _ = tf.add(e_tensor, f_tensor, name="g")
    g = gde.Graph(tf_graph)

    sgv0 = gde.sgv(g["a"], g["b"], g["c"])
    sgv1 = gde.sgv(g["e"], g["f"])

    gde.swap_outputs(sgv0, sgv1)
    self.assertTrue(
        gde.OpMatcher("g").input_ops(
            "a", gde.OpMatcher("c").input_ops("a", "b"))(g["g"]))
    self.assertTrue(gde.OpMatcher("d").input_ops("e", "f")(g["d"]))


if __name__ == "__main__":
  unittest.main()
