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

import re

import tensorflow as tf
import unittest

import graph_def_editor as gde


class SelectTest(unittest.TestCase):

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

  def test_regex(self):
    """Test for ge.can_be_regex and ge.make_regex."""
    self.assertTrue(gde.can_be_regex("foo"))
    self.assertTrue(gde.can_be_regex(re.compile("foo")))
    regex = re.compile("foo")
    self.assertIs(gde.make_regex(regex), regex)

  def test_get_input_output_ts(self):
    """Test for ge._get_input_ts abd ge._get_output_ts."""
    self.assertEqual(len(gde.select._get_input_ts(self.graph)), 6)
    self.assertEqual(len(gde.select._get_output_ts(self.graph)), 8)

  def test_get_filter(self):
    """Test for various filtering operations on ts ops."""
    # TODO(fkp): parameterize
    self.assertEqual(len(gde.filter_ops(self.graph, True)), 8)
    self.assertEqual(
        len(gde.filter_ops(self.graph,
                                        lambda op: op.op_type == "Const")), 3)
    self.assertEqual(
        len(gde.filter_ops(self.graph, lambda op: op.op_type == "Add")), 5)
    self.assertEqual(
        len(gde.filter_ops_from_regex(self.graph, r"^.*\b[abc]$")), 3)

    self.assertEqual(len(gde.filter_ts(self.graph, True)), 8)
    self.assertEqual(
        len(gde.filter_ts_from_regex(self.graph, r"^.*/[fgh]:\d$")), 3)

    self.assertEqual(len(gde.get_name_scope_ops(self.graph, "foo/")), 7)
    self.assertEqual(len(gde.get_name_scope_ops(self.graph, "foo/bar")), 4)

  def test_get_ops_ios(self):
    """Test for ge.get_ops_ios."""
    control_outputs = gde.util.ControlOutputs(self.graph)
    self.assertEqual(
        len(gde.get_ops_ios(self.h.op, control_ios=control_outputs)), 3)
    self.assertEqual(len(gde.get_ops_ios(self.h.op)), 2)
    self.assertEqual(
        len(gde.get_ops_ios(self.c.op, control_ios=control_outputs)), 6)
    self.assertEqual(len(gde.get_ops_ios(self.c.op)), 5)

  def test_compute_boundary_ts_0(self):
    """Test for ge.compute_boundary_ts."""
    input_ts, output_ts, inside_ts = gde.compute_boundary_ts(self.g.op)
    self.assertEqual(list(input_ts), [self.c, self.a])
    self.assertEqual(list(output_ts), [self.g])
    self.assertEqual(list(inside_ts), [])

  def test_compute_boundary_ts_1(self):
    """Test for ge.compute_boundary_ts."""
    input_ts, output_ts, inside_ts = gde.compute_boundary_ts(
        [self.g.op, self.h.op])
    self.assertEqual(list(input_ts), [self.c, self.a, self.f])
    self.assertEqual(list(output_ts), [self.h])
    self.assertEqual(list(inside_ts), [self.g])

  def test_compute_boundary_ts_2(self):
    """Test for ge.compute_boundary_ts."""
    tf_graph = tf.Graph()
    with tf_graph.as_default():
      a_tensor = tf.constant(1, name="a")
      b_tensor = tf.constant(1, name="b")
      c_tensor = tf.add(a_tensor, b_tensor, name="c")
      _ = a_tensor + c_tensor

    g = gde.Graph(tf_graph)
    input_ts, output_ts, inside_ts = gde.compute_boundary_ts([g["a"], g["c"]])
    self.assertEqual(list(input_ts), [g["b"].output(0)])
    self.assertEqual(list(output_ts), [g["a"].output(0), g["c"].output(0)])
    self.assertEqual(list(inside_ts), [g["a"].output(0)])

  def test_get_within_boundary_ops_0(self):
    """Test for test_get_within_boundary_ops."""
    control_outputs = gde.util.ControlOutputs(self.graph)
    ops = gde.get_within_boundary_ops(
        ops=self.graph,
        seed_ops=self.f.op,
        boundary_ops=[self.c.op, self.h.op],
        inclusive=False,
        control_ios=control_outputs)
    self.assertEqual(len(ops), 3)

  def test_get_within_boundary_ops_1(self):
    """Test for ge.test_get_within_boundary_ops."""
    ops = gde.get_within_boundary_ops(
        ops=self.graph, seed_ops=self.h.op, boundary_ops=[self.f.op, self.g.op])
    self.assertEqual(len(ops), 3)

  def test_get_walks_intersection(self):
    """Test for ge.get_walks_intersection_ops."""
    ops = gde.get_walks_intersection_ops([self.c.op], [self.g.op])
    self.assertEqual(len(ops), 2)

    ops = gde.get_walks_intersection_ops([self.a.op], [self.f.op])
    self.assertEqual(len(ops), 3)
    self.assertTrue(self.a.op in ops)
    self.assertTrue(self.c.op in ops)
    self.assertTrue(self.f.op in ops)

    within_ops = [self.a.op, self.f.op]
    ops = gde.get_walks_intersection_ops(
        [self.a.op], [self.f.op], within_ops=within_ops)
    self.assertEqual(len(ops), 0)

    def within_ops_fn(op):
      return op in [self.a.op, self.f.op]
    ops = gde.get_walks_intersection_ops(
        [self.a.op], [self.f.op], within_ops_fn=within_ops_fn)
    self.assertEqual(len(ops), 0)

  def test_get_walks_union(self):
    """Test for ge.get_walks_union_ops."""
    ops = gde.get_walks_union_ops([self.f.op], [self.g.op])
    self.assertEqual(len(ops), 6)

    ops = gde.get_walks_union_ops([self.a.op], [self.f.op])
    self.assertEqual(len(ops), 8)

    within_ops = [self.a.op, self.c.op, self.d.op, self.f.op]
    ops = gde.get_walks_union_ops([self.a.op], [self.f.op],
                                               within_ops=within_ops)
    self.assertEqual(len(ops), 4)
    self.assertTrue(self.b.op not in ops)

    def within_ops_fn(op):
      return op in [self.a.op, self.c.op, self.f.op]

    ops = gde.get_walks_union_ops([self.a.op], [self.f.op],
                                               within_ops_fn=within_ops_fn)
    self.assertEqual(len(ops), 3)
    self.assertTrue(self.b.op not in ops)
    self.assertTrue(self.d.op not in ops)

  def test_select_ops(self):
    parameters = (
        (("^foo/",), 7),
        (("^foo/bar/",), 4),
        (("^foo/bar/", "a"), 5),
    )
    for param, length in parameters:
      ops = gde.select_ops(*param, graph=self.graph)
      self.assertEqual(len(ops), length)

  def test_select_ts(self):
    parameters = (
        (".*:0", 8),
        (r".*/bar/\w+:0", 4),
    )
    for regex, length in parameters:
      ts = gde.select_ts(regex, graph=self.graph)
      self.assertEqual(len(ts), length)

  def test_select_ops_and_ts(self):
    parameters = (
        (("^foo/.*",), 7, 0),
        (("^foo/.*", "(?#ts)^foo/bar/.*"), 7, 4),
    )
    for param, l0, l1 in parameters:
      ops, ts = gde.select_ops_and_ts(*param, graph=self.graph)
      self.assertEqual(len(ops), l0)
      self.assertEqual(len(ts), l1)

  def test_forward_walk_ops(self):
    seed_ops = [self.a.op, self.d.op]
    # Include all ops except for self.g.op
    within_ops = [
        x.op for x in [self.a, self.b, self.c, self.d, self.e, self.f, self.h]
    ]

    # For the fn, exclude self.e.op.
    def within_ops_fn(op):
      return op not in (self.e.op,)
    stop_at_ts = (self.f,)

    # No b.op since it's an independent source node.
    # No g.op from within_ops.
    # No e.op from within_ops_fn.
    # No h.op from stop_at_ts and within_ops.
    ops = gde.select.get_forward_walk_ops(
        seed_ops,
        inclusive=True,
        within_ops=within_ops,
        within_ops_fn=within_ops_fn,
        stop_at_ts=stop_at_ts)
    self.assertEqual(
        set(ops), {self.a.op, self.c.op, self.d.op, self.f.op })

    # Also no a.op and d.op when inclusive=False
    ops = gde.select.get_forward_walk_ops(
        seed_ops,
        inclusive=False,
        within_ops=within_ops,
        within_ops_fn=within_ops_fn,
        stop_at_ts=stop_at_ts)
    self.assertEqual(set(ops), {self.c.op, self.f.op})

    # Not using within_ops_fn adds e.op.
    ops = gde.select.get_forward_walk_ops(
        seed_ops,
        inclusive=False,
        within_ops=within_ops,
        stop_at_ts=stop_at_ts)
    self.assertEqual(set(ops), {self.c.op, self.e.op, self.f.op})

    # Not using stop_at_ts adds back h.op.
    ops = gde.select.get_forward_walk_ops(
        seed_ops, inclusive=False, within_ops=within_ops)
    self.assertEqual(
        set(ops), {self.c.op, self.e.op, self.f.op, self.h.op})

    # Starting just form a (the tensor, not op) omits a, b, d.
    ops = gde.select.get_forward_walk_ops([self.a], inclusive=True)
    self.assertEqual(
        set(ops), {self.c.op, self.e.op, self.f.op, self.g.op, self.h.op})

  def test_backward_walk_ops(self):
    seed_ops = [self.h.op]
    # Include all ops except for self.g.op
    within_ops = [
        x.op for x in [self.a, self.b, self.c, self.d, self.e, self.f, self.h]
    ]

    # For the fn, exclude self.c.op.
    def within_ops_fn(op):
      return op not in (self.c.op,)
    stop_at_ts = (self.f,)

    # Backward walk only includes h since we stop at f and g is not within.
    ops = gde.select.get_backward_walk_ops(
        seed_ops,
        inclusive=True,
        within_ops=within_ops,
        within_ops_fn=within_ops_fn,
        stop_at_ts=stop_at_ts)
    self.assertEqual(set(ops), {self.h.op})

    # If we do inclusive=False, the result is empty.
    ops = gde.select.get_backward_walk_ops(
        seed_ops,
        inclusive=False,
        within_ops=within_ops,
        within_ops_fn=within_ops_fn,
        stop_at_ts=stop_at_ts)
    self.assertEqual(set(ops), set())

    # Removing stop_at_fs adds f.op, d.op.
    ops = gde.select.get_backward_walk_ops(
        seed_ops,
        inclusive=True,
        within_ops=within_ops,
        within_ops_fn=within_ops_fn)
    self.assertEqual(set(ops), {self.d.op, self.f.op, self.h.op})

    # Not using within_ops_fn adds back ops for a, b, c.
    ops = gde.select.get_backward_walk_ops(
        seed_ops, inclusive=True, within_ops=within_ops)
    self.assertEqual(
        set(ops),
        {self.a.op, self.b.op, self.c.op, self.d.op, self.f.op, self.h.op})

    # Vanially backward search via self.h.op includes everything except e.op.
    ops = gde.select.get_backward_walk_ops(seed_ops, inclusive=True)
    self.assertEqual(
        set(ops),
        {self.a.op, self.b.op, self.c.op, self.d.op, self.f.op, self.g.op,
         self.h.op})


if __name__ == "__main__":
  unittest.main()
