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

  def test_graph_collection_types(self):

    # Build a graph with NodeList that has an operation and tensor,
    # and ByteList with variable
    tf_g = tf.Graph()
    with tf_g.as_default():
      y_ = tf.placeholder(tf.int64, [None])
      x = tf.get_variable("x", [1])
      with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_,
                                                               logits=x)
      with tf.name_scope('adam_optimizer'):
        _ = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    g = gde.Graph(tf_g)

    keys = g.get_all_collection_keys()

    # Check that loss tensor added to collection
    self.assertIn('losses', keys)
    t = g.get_tensor_by_name("loss/sparse_softmax_cross_entropy_loss/value:0")
    self.assertIn('losses', t.collection_names)

    # Check that variable added to collection
    self.assertIn('variables', keys)
    v = g.get_variable_by_name('x:0')
    self.assertIn('variables', v.collection_names)

    # Check that op added to collection
    self.assertIn('train_op', keys)
    n = g.get_node_by_name("adam_optimizer/Adam")
    self.assertIn('train_op', n.collection_names)

  def test_collection_roundtrip_savedmodel(self):
    tf_g = tf.Graph()
    with tf_g.as_default():
      x = tf.placeholder(dtype=tf.float32, shape=[])
      y = tf.placeholder(dtype=tf.float32, shape=[])
      w = tf.Variable([1.0, 2.0], name="w")
      c = tf.constant(0.0)
      tf.add_to_collection('tensors', c)
      y_model = tf.multiply(x, w[0]) + w[1] + c

      error = tf.square(y - y_model)
      train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)
      model = tf.global_variables_initializer()

      with tf.Session() as sess:
        sess.run(model)
        sess.run(train_op, feed_dict={x: 0.5, y: 1.0})
        sess.run(w)
        model_dir = self.temp_dir + "/saved_model"
        tf.saved_model.simple_save(sess, model_dir,
                                   inputs={"in": x},
                                   outputs={"out": error})

    expected_collections = ['variables', 'tensors', 'train_op']

    # Checking for initial collections
    for name in expected_collections:
      self.assertIn(name, tf_g.collections)

    # Load tf savedmodel with gde
    g = gde.saved_model_to_graph(model_dir)

    # Check collections are loaded from tf savedmodel
    collections = g.get_all_collection_keys()
    for name in expected_collections:
      self.assertIn(name, collections)

    # Check collections are assigned when loaded
    w_gde = g.get_variable_by_name(w.name)
    self.assertIn('variables', w_gde.collection_names)
    c_gde = g.get_tensor_by_name(c.name)
    self.assertIn('tensors', c_gde.collection_names)
    train_op_gde = g.get_node_by_name(train_op.name)
    self.assertIn('train_op', train_op_gde.collection_names)

    # Use gde to write savedmodel
    second_model_dir = self.temp_dir + "/saved_model_after"
    g.to_saved_model(second_model_dir)

    # Load gde savedmodel in tf session
    after_tf_g = tf.Graph()
    with after_tf_g.as_default():
      with tf.Session() as sess:
        tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING],
                            second_model_dir)

    # Checking collections loaded back from gde savedmodel
    for name in expected_collections:
      self.assertIn(name, after_tf_g.collections)

    # Check that collections have expected contents after tf.saved_model.load
    variable_collection_names = [v.name for v in after_tf_g.get_collection('variables')]
    self.assertIn(w_gde.name, variable_collection_names)
    tensors_collection_names = [t.name for t in after_tf_g.get_collection('tensors')]
    self.assertIn(c.name, tensors_collection_names)
    op_collection_names = [o.name for o in after_tf_g.get_collection('train_op')]
    self.assertIn(train_op.name, op_collection_names)

  def test_node_collection_type_unique(self):
    g = gde.Graph()
    a = g.add_node("a", "a_op")
    a.set_outputs_from_pairs([(tf.int32, tf.TensorShape([]))])
    a.add_to_collection("mixed_collection")
    b = g.add_node("b", "b_op")
    b.set_outputs_from_pairs([(tf.int32, tf.TensorShape([]))])
    t = b.outputs[0]
    t.add_to_collection("mixed_collection")
    with self.assertRaisesRegex(TypeError, "Node collections cannot be Nodes and Tensors.*"):
      g.get_collection_by_name("mixed_collection")


if __name__ == "__main__":
  unittest.main()
