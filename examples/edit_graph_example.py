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

"""Simple GraphEditor example.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import graph_def_editor as gde

FLAGS = tf.flags.FLAGS


def main(_):
  # Create a graph
  tf_g = tf.Graph()
  with tf_g.as_default():
    a = tf.constant(1.0, shape=[2, 3], name="a")
    b = tf.constant(2.0, shape=[2, 3], name="b")
    c = tf.add(
        tf.placeholder(dtype=np.float32),
        tf.placeholder(dtype=np.float32),
        name="c")

  # Serialize the graph
  g = gde.Graph(g.as_graph_def())
  print("Before:\n{}".format(g.to_graph_def()))

  # Modify the graph
  gde.swap_inputs(g[c.op.name], [g[a.name], g[b.name]])

  print("After:\n{}".format(g.to_graph_def()))

  # Reconstitute the modified serialized graph as TensorFlow graph...
  with g.to_tf_graph().as_default():
    # ...and print the value of c
    with tf.Session() as sess:
      res = sess.run(c.name)
      print(res)


if __name__ == "__main__":
  tf.app.run()
