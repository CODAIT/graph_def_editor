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


class MatchTest(unittest.TestCase):

  def setUp(self):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
      a = tf.constant([1., 1.], shape=[2], name="a")
      with tf.name_scope("foo"):
        b = tf.constant([2., 2.], shape=[2], name="b")
        c = tf.add(a, b, name="c")
        d = tf.constant([3., 3.], shape=[2], name="d")
        with tf.name_scope("bar"):
          _ = tf.add(c, d, name="e")
          f = tf.add(c, d, name="f")
          g = tf.add(c, a, name="g")
          with tf.control_dependencies([c.op]):
            _ = tf.add(f, g, name="h")

    self.graph = gde.Graph(tf_graph)
    self.f_op = self.graph[f.op.name]

  def test_simple_match(self):
    self.assertTrue(gde.OpMatcher("^.*/f$")(self.f_op))
    self.assertTrue(
        gde.OpMatcher("^.*/f$").input_ops("^.*/c$", "^.*/d$")(self.f_op))
    self.assertTrue(
        gde.OpMatcher("^.*/f$").input_ops(True, "^.*/d$")(self.f_op))
    self.assertTrue(
        gde.OpMatcher("^.*/f$").input_ops(
            gde.op_type("Add"), gde.op_type("Const"))(self.f_op))
    self.assertTrue(
        gde.OpMatcher("^.*/f$").input_ops("^.*/c$", "^.*/d$")
        .output_ops(gde.OpMatcher("^.*/h$")
                    .control_input_ops("^.*/c$"))(self.f_op))
    self.assertTrue(
        gde.OpMatcher("^.*/f$").input_ops("^.*/c$", "^.*/d$").output_ops(
            gde.OpMatcher("^.*/h$").control_input_ops("^.*/c$")
            .output_ops([]))(self.f_op))


if __name__ == "__main__":
  unittest.main()
