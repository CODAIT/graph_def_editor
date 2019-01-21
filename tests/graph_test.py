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

"""
Tests for graph.py in the GraphDef Editor
"""

import unittest
import tensorflow as tf
import numpy as np
import shutil
import tempfile


import graph_def_editor as gde


class GraphTest(unittest.TestCase):

  def setUp(self):
    # Create a temporary directory for SavedModel files.
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    # Remove the directory after the test.
    # Comment out this line to prevent deleting temps.
    shutil.rmtree(self.temp_dir)
    pass  # In case previous line gets commented out

  def test_import_saved_model(self):
    tf_g = tf.Graph()
    with tf_g.as_default():
      input_tensor = tf.placeholder(dtype=tf.int32, shape=[],
                                    name="Input")
      result_tensor = input_tensor + 42

      model_dir = self.temp_dir + "/saved_model"
      with tf.Session() as sess:
        tf.saved_model.simple_save(sess, model_dir,
                                   inputs={"in": input_tensor},
                                   outputs={"out": result_tensor})

    g = gde.saved_model_to_graph(model_dir)
    with g.to_tf_graph().as_default():
      with tf.Session() as sess:
        result = sess.run(result_tensor.name, {input_tensor.name: 1})
    self.assertEqual(result, 43)

  def test_export_saved_model_no_vars(self):
    """Generate a graph in memory with no variables and export as SavedModel
    (with empty checkpoint)"""
    tf_g = tf.Graph()
    with tf_g.as_default():
      input_tensor = tf.placeholder(dtype=tf.int32, shape=[],
                                    name="Input")
      result_tensor = input_tensor + 42
    g = gde.Graph(tf_g)
    model_dir = self.temp_dir + "/saved_model"
    g.to_saved_model(model_dir)

    # Load the model we just saved and do a test run
    after_tf_g = tf.Graph()
    with after_tf_g.as_default():
      with tf.Session() as sess:
        tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING],
                            model_dir)
        result = sess.run(result_tensor.name, {input_tensor.name: 1})
        self.assertEqual(result, 43)

  def test_export_saved_model_with_var(self):
    """Import a SavedModel with a variable, modify the resulting graph,
    and write it out as a second SavedModel"""
    tf_g = tf.Graph()
    with tf_g.as_default():
      input_tensor = tf.placeholder(dtype=tf.int32, shape=[],
                                    name="Input")
      var_tensor = tf.Variable(initial_value=42, name="FortyTwo")
      result_tensor = input_tensor + var_tensor

      with tf.Session() as sess:
        sess.run(var_tensor.initializer)
        model_dir = self.temp_dir + "/saved_model"
        tf.saved_model.simple_save(sess, model_dir,
                                   inputs={"in": input_tensor},
                                   outputs={"out": result_tensor})

    g = gde.saved_model_to_graph(model_dir)

    # Verify that the import went ok
    with g.to_tf_graph().as_default():
      with tf.Session() as sess:
        sess.run(var_tensor.initializer.name)
        result = sess.run(result_tensor.name, {input_tensor.name: 1})
    self.assertEqual(result, 43)

    # Now rewrite plus to minus.
    result_op = g.get_node_by_name(result_tensor.op.name)
    result_op.change_op_type("Sub")

    second_model_dir = self.temp_dir + "/saved_model_after"
    g.to_saved_model(second_model_dir)

    after_tf_g = tf.Graph()
    with after_tf_g.as_default():
      with tf.Session() as sess:
        tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING],
                            second_model_dir)
        result = sess.run(result_tensor.name, {input_tensor.name: 1})
        self.assertEqual(result, -41)


if __name__ == "__main__":
  unittest.main()
