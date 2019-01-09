# Coypright 2018 IBM. All Rights Reserved.
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

"""Simple example of the GraphDef Editor.

To run this example from the root of the project, type:
   PYTHONPATH=$PWD env/bin/python examples/edit_graph_example.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import graph_def_editor as gde
import textwrap

FLAGS = tf.flags.FLAGS


def _indent(s):
  return textwrap.indent(str(s), "    ")


def main(_):
  # Create a graph
  tf_g = tf.Graph()
  with tf_g.as_default():
    a = tf.constant(1.0, shape=[2, 3], name="a")
    c = tf.add(
        tf.placeholder(dtype=np.float32),
        tf.placeholder(dtype=np.float32),
        name="c")

  # Serialize the graph
  g = gde.Graph(tf_g.as_graph_def())
  print("Before:\n{}".format(_indent(g.to_graph_def())))

  # Modify the graph.
  # In this case we replace the two input placeholders with constants.
  # One of the constants (a) is a node that was in the original graph.
  # The other one (b) we create here.
  b = gde.make_const(g, "b", np.full([2, 3], 2.0, dtype=np.float32))
  gde.swap_inputs(g[c.op.name], [g[a.name], b.output(0)])

  print("After:\n{}".format(_indent(g.to_graph_def())))

  # Reconstitute the modified serialized graph as TensorFlow graph...
  with g.to_tf_graph().as_default():
    # ...and print the value of c, which should be 2x3 matrix of 3.0's
    with tf.Session() as sess:
      res = sess.run(c.name)
      print("Result is:\n{}".format(_indent(res)))


if __name__ == "__main__":
  tf.app.run()
