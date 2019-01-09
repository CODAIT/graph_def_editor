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


class SubgraphTest(unittest.TestCase):

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

  def test_subgraph(self):
    sgv = gde.sgv(self.graph)
    self.assertEqual(list(sgv.outputs), [self.e, self.h])
    self.assertEqual(list(sgv.inputs), [])
    self.assertEqual(len(sgv.ops), 8)

    sgv = gde.sgv(self.f.op, self.g.op)
    self.assertEqual(list(sgv.outputs), [self.f, self.g])
    self.assertEqual(list(sgv.inputs), [self.c, self.d, self.a])

    sgv = gde.sgv_scope("foo/bar", graph=self.graph)
    self.assertEqual(
        list(sgv.ops), [self.e.op, self.f.op, self.g.op, self.h.op])

  def test_subgraph_remap(self):
    sgv = gde.sgv(self.c.op)
    self.assertEqual(list(sgv.outputs), [self.c])
    self.assertEqual(list(sgv.inputs), [self.a, self.b])

    sgv = gde.sgv(self.c.op).remap([self.a], [0, self.c])
    self.assertEqual(list(sgv.outputs), [self.c, self.c])
    self.assertEqual(list(sgv.inputs), [self.a])

    sgv = sgv.remap_outputs_to_consumers()
    self.assertEqual(list(sgv.outputs), [self.c, self.c, self.c])
    sgv = sgv.remap_outputs_make_unique()
    self.assertEqual(list(sgv.outputs), [self.c])

    sgv = sgv.remap(new_input_indices=[], new_output_indices=[])
    self.assertEqual(len(sgv.inputs), 0)
    self.assertEqual(len(sgv.outputs), 0)
    sgv = sgv.remap_default()
    self.assertEqual(list(sgv.outputs), [self.c])
    self.assertEqual(list(sgv.inputs), [self.a, self.b])

  def test_remove_unused_ops(self):
    sgv = gde.sgv(self.graph)
    self.assertEqual(list(sgv.outputs), [self.e, self.h])
    self.assertEqual(len(sgv.ops), 8)

    sgv = sgv.remap_outputs(new_output_indices=[1]).remove_unused_ops()
    self.assertEqual(list(sgv.outputs), [self.h])
    self.assertEqual(len(sgv.ops), 7)


if __name__ == "__main__":
  test.main()
