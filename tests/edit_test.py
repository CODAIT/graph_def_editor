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
"""Tests for gde.edit"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import unittest

import graph_def_editor as gde


class EditTest(unittest.TestCase):
  """edit module test.

  Generally the tests are in two steps:
  - modify an existing graph.
  - then make sure it has the expected topology using the graph matcher.
  """

  # TODO(frreiss): Merge duplicate setup code across test cases
  def setUp(self):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
      a = tf.constant([1., 1.], shape=[2], name="a")
      with tf.name_scope("foo"):
        b = tf.constant([2., 2.], shape=[2], name="b")
        c = tf.add(a, b, name="c")
        d = tf.constant([3., 3.], shape=[2], name="d")
        with tf.name_scope("bar"):
          e = tf.add(c, d, name="e")
          f = tf.add(c, d, name="f")
          g = tf.add(c, a, name="g")
          with tf.control_dependencies([c.op]):
            h = tf.add(f, g, name="h")
    self.graph = gde.Graph(tf_graph)
    self.a = self.graph.get_tensor_by_name(a.name)
    self.b = self.graph.get_tensor_by_name(b.name)
    self.c = self.graph.get_tensor_by_name(c.name)
    self.d = self.graph.get_tensor_by_name(d.name)
    self.e = self.graph.get_tensor_by_name(e.name)
    self.f = self.graph.get_tensor_by_name(f.name)
    self.g = self.graph.get_tensor_by_name(g.name)
    self.h = self.graph.get_tensor_by_name(h.name)

  def test_detach(self):
    """Test for ge.detach."""
    sgv = gde.sgv(self.c.op, self.a.op)
    control_outputs = gde.ControlOutputs(self.graph)
    gde.detach(sgv, control_ios=control_outputs)
    # make sure the detached graph is as expected.
    self.assertTrue(
        gde.OpMatcher("^foo/c$").input_ops("a", "geph__b_0")(self.c.op))

  def test_connect(self):
    """Test for gde.connect."""
    # Original code:
    # with self.graph.as_default():
    #   x = constant_op.constant([1., 1.], shape=[2], name="x")
    #   y = constant_op.constant([2., 2.], shape=[2], name="y")
    #   z = math_ops.add(x, y, name="z")
    x = gde.make_const(self.graph, "x", np.array([1., 1.], dtype=np.float32))
    y = gde.make_const(self.graph, "y", np.array([2., 2.], dtype=np.float32))
    z = self.graph.add_node("z", "Add")
    z.add_attr("T", tf.float32)
    z.set_inputs([x.outputs[0], y.outputs[0]])
    z.infer_outputs()

    sgv = gde.sgv(x, y, z)
    gde.connect(sgv, gde.sgv(self.e.op).remap_inputs([0]))
    self.assertTrue(
        gde.OpMatcher("^foo/bar/e$").input_ops("^z$", "foo/d$")(self.e.op))

  def test_bypass(self):
    """Test for ge.bypass."""
    gde.bypass(gde.sgv(self.f.op).remap_inputs([0]))
    self.assertTrue(
        gde.OpMatcher("^foo/bar/h$").input_ops("^foo/c$", "foo/bar/g$")(
            self.h.op))


if __name__ == "__main__":
  unittest.main()
