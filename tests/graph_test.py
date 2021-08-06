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
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
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

  def build_graph(self):
    tf_g = tf.Graph()
    with tf_g.as_default():
      a = tf.constant(1, name="a")
      b = tf.constant(2, name="b")
      c = tf.constant(10, name="c")
      add_res = tf.add(a, b, name="add")
      res = tf.multiply(add_res, c, name="mult")
    g = gde.Graph(g=tf_g)
    return g

  def build_graph_with_function(self):
    """Builds a tf graph for function (x + y) * 10.0 ."""
    @tf.function
    def multiplier_function(v):
      return tf.constant(10.0, name="function_multiplier") * v

    tf_g = tf.Graph()
    with tf_g.as_default():
      x = tf.placeholder(name="x", dtype=tf.float32, shape=[])
      y = tf.placeholder(name="y", dtype=tf.float32, shape=[])
      result_op = tf.add(x, y, name="add")
      func_call_op = multiplier_function(result_op)
      _ = tf.identity(func_call_op, name="output")
    return gde.Graph(g=tf_g)

  def build_graph_with_nested_function_call(self):
    """Builds a tf graph for function (x + y) * 10.0 ."""
    @tf.function
    def adder_function(a, b):
      return a + b

    @tf.function
    def multiplier_function(a, b):
      v = adder_function(a, b)
      return tf.constant(10.0, name="function_multiplier") * v

    tf_g = tf.Graph()
    with tf_g.as_default():
      x = tf.placeholder(name="x", dtype=tf.float32, shape=[])
      y = tf.placeholder(name="y", dtype=tf.float32, shape=[])
      func_call_op = multiplier_function(x, y)
      _ = tf.identity(func_call_op, name="output")
    return gde.Graph(g=tf_g)

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

  def test_nodes_iterator(self):
    g = self.build_graph_with_function()
    self.assertEqual(
        {g.get_node_by_name("x"),
         g.get_node_by_name("y"),
         g.get_node_by_name("add"),
         g.get_node_by_name("PartitionedCall"),
         g.get_node_by_name("output")},
        set(g.nodes_iterator()))

  def test_nodes_iterator_predicate(self):
    g = self.build_graph_with_function()
    self.assertEqual(
        {g.get_node_by_name("x"),
         g.get_node_by_name("y")},
        set(g.nodes_iterator(predicate=lambda n: n.op_type == "Placeholder")))

  def test_nodes_iterator_iterate_functions(self):
    g = self.build_graph_with_function()
    f = g.get_function_graph_by_name(g.function_names[0])
    self.assertEqual(
        {g.get_node_by_name("x"),
         g.get_node_by_name("y"),
         g.get_node_by_name("add"),
         g.get_node_by_name("PartitionedCall"),
         g.get_node_by_name("output"),
         f.get_node_by_name("function_multiplier"),
         f.get_node_by_name("mul"),
         f.get_node_by_name("Identity"),
         f.get_node_by_name("v")},
        set(g.nodes_iterator(iterate_functions=True)))

  def test_breadth_first_visitor(self):
    g = self.build_graph()
    nodes_in_bfs = []
    def visit(node):
      nodes_in_bfs.append(node)
    def visit_with_break(node):
      nodes_in_bfs.append(node)
      return True
    g.breadth_first_visitor(visit)
    self.assertEqual(
        [g.get_node_by_name("a"),
         g.get_node_by_name("b"),
         g.get_node_by_name("c"),
         g.get_node_by_name("add"),
         g.get_node_by_name("mult")],
        nodes_in_bfs)

    nodes_in_bfs = []
    g.breadth_first_visitor(visit,
                            starting_nodes=[g.get_node_by_name("a")])
    self.assertEqual(
        [g.get_node_by_name("a"),
         g.get_node_by_name("add"),
         g.get_node_by_name("mult")],
        nodes_in_bfs)

    nodes_in_bfs = []
    g.breadth_first_visitor(visit,
                            starting_nodes=[g.get_node_by_name("c")])
    self.assertEqual(
        [g.get_node_by_name("c"),
         g.get_node_by_name("mult")],
        nodes_in_bfs)

    nodes_in_bfs = []
    g.breadth_first_visitor(visit_with_break,
                            starting_nodes=[g.get_node_by_name("c")])
    self.assertEqual(
        [g.get_node_by_name("c")],
        nodes_in_bfs)

  def test_breadth_first_visitor_iterate_functions(self):
    g = self.build_graph_with_function()
    nodes_in_bfs = []
    def visit(node):
      nodes_in_bfs.append(node)
    g.breadth_first_visitor(
        visit,
        starting_nodes=[g.get_node_by_name("x"), g.get_node_by_name("y")])
    self.assertEqual(
        [g.get_node_by_name("x"),
         g.get_node_by_name("y"),
         g.get_node_by_name("add"),
         g.get_node_by_name("PartitionedCall"),
         g.get_node_by_name("output")],
        nodes_in_bfs)

    nodes_in_bfs = []
    f = g.get_function_graph_by_name(g.function_names[0])
    g.breadth_first_visitor(
        visit,
        starting_nodes=[g.get_node_by_name("x"), g.get_node_by_name("y")],
        iterate_functions=True)
    self.assertEqual(
        [g.get_node_by_name("x"),
         g.get_node_by_name("y"),
         g.get_node_by_name("add"),
         g.get_node_by_name("PartitionedCall"),
         f.get_node_by_name("mul"),
         f.get_node_by_name("Identity"),
         g.get_node_by_name("output")],
        nodes_in_bfs)

  def test_breadth_first_visitor_escape_functions(self):
    g = self.build_graph_with_function()
    nodes_in_bfs = []
    def visit(node):
      nodes_in_bfs.append(node)
    f = g.get_function_graph_by_name(g.function_names[0])
    g.breadth_first_visitor(
        visit,
        starting_nodes=[f.get_node_by_name("function_multiplier")])
    self.assertEqual(
        [f.get_node_by_name("function_multiplier"),
         f.get_node_by_name("mul"),
         f.get_node_by_name("Identity")],
        nodes_in_bfs)

    nodes_in_bfs = []
    f = g.get_function_graph_by_name(g.function_names[0])
    g.breadth_first_visitor(
        visit,
        starting_nodes=[f.get_node_by_name("function_multiplier")],
        escape_functions=True)
    self.assertEqual(
        [f.get_node_by_name("function_multiplier"),
         f.get_node_by_name("mul"),
         f.get_node_by_name("Identity"),
         g.get_node_by_name("output")],
        nodes_in_bfs)

  def test_breadth_first_visitor_escape_nested_functions(self):
    g = self.build_graph_with_nested_function_call()
    nodes_in_bfs = []
    def visit(node):
      nodes_in_bfs.append(node)

    nodes_in_bfs = []
    f = g.get_function_graph_by_name(g.function_names[0])
    g.breadth_first_visitor(
        visit,
        starting_nodes=[f.get_node_by_name("function_multiplier")],
        iterate_functions=True,
        escape_functions=True)
    self.assertEqual(
        [f.get_node_by_name("function_multiplier"),
         f.get_node_by_name("mul"),
         f.get_node_by_name("Identity"),
         g.get_node_by_name("output")],
        nodes_in_bfs)

  def test_breadth_first_visitor_escape_nested_functions(self):
    g = self.build_graph_with_nested_function_call()
    nodes_in_bfs = []
    def visit(node):
      nodes_in_bfs.append(node)

    add_node = list(g.nodes_iterator(lambda n:n.name=='add', iterate_functions=True))[0]
    multiplier_function_name = g.get_node_by_name("x").outputs[0].consumers()[0].get_attr('f').name
    multiplier_function_graph = g.get_function_graph_by_name(multiplier_function_name)
    adder_function_graph = add_node.graph
    nodes_in_bfs = []
    g.breadth_first_visitor(
        visit,
        starting_nodes=[add_node],
        iterate_functions=True,
        escape_functions=True)
    self.assertEqual(
        [add_node,
         add_node.graph.get_node_by_name("Identity"),
         multiplier_function_graph.get_node_by_name("mul"),
         multiplier_function_graph.get_node_by_name("Identity"),
         g.get_node_by_name("output")],
        nodes_in_bfs)

  def test_backwards_breadth_first_visitor(self):
    g = self.build_graph()
    nodes_in_backwards_bfs = []
    def visit(node):
      nodes_in_backwards_bfs.append(node)
    def visit_with_break(node):
      nodes_in_backwards_bfs.append(node)
      return True
    g.backwards_breadth_first_visitor(
        visit,
        starting_nodes=[g.get_node_by_name("mult")])
    self.assertEqual(
        [g.get_node_by_name("mult"),
         g.get_node_by_name("add"),
         g.get_node_by_name("c"),
         g.get_node_by_name("a"),
         g.get_node_by_name("b")],
        nodes_in_backwards_bfs)

    nodes_in_backwards_bfs = []
    g.backwards_breadth_first_visitor(
        visit,
        starting_nodes=[g.get_node_by_name("add")])
    self.assertEqual(
        [g.get_node_by_name("add"),
         g.get_node_by_name("a"),
         g.get_node_by_name("b")],
        nodes_in_backwards_bfs)

    nodes_in_backwards_bfs = []
    g.backwards_breadth_first_visitor(
        visit_with_break,
        starting_nodes=[g.get_node_by_name("add")])
    self.assertEqual(
        [g.get_node_by_name("add")],
        nodes_in_backwards_bfs)

  def test_backwards_breadth_first_visitor_iterate_functions(self):
    g = self.build_graph_with_function()
    nodes_in_backwards_bfs = []
    def visit(node):
      nodes_in_backwards_bfs.append(node)
    g.backwards_breadth_first_visitor(
        visit,
        starting_nodes=[g.get_node_by_name("output")])
    self.assertEqual(
        [g.get_node_by_name("output"),
         g.get_node_by_name("PartitionedCall"),
         g.get_node_by_name("add"),
         g.get_node_by_name("x"),
         g.get_node_by_name("y")],
        nodes_in_backwards_bfs)

    nodes_in_backwards_bfs = []
    f = g.get_function_graph_by_name(g.function_names[0])
    g.backwards_breadth_first_visitor(
        visit,
        starting_nodes=[g.get_node_by_name("output")],
        iterate_functions=True)
    self.assertEqual(
        [g.get_node_by_name("output"),
         g.get_node_by_name("PartitionedCall"),
         f.get_node_by_name("Identity"),
         f.get_node_by_name("mul"),
         f.get_node_by_name("function_multiplier"),
         g.get_node_by_name("add"),
         g.get_node_by_name("x"),
         g.get_node_by_name("y")],
        nodes_in_backwards_bfs)

  def test_backwards_breadth_first_visitor_escape_functions(self):
    g = self.build_graph_with_function()
    nodes_in_backwards_bfs = []
    def visit(node):
      nodes_in_backwards_bfs.append(node)
    f = g.get_function_graph_by_name(g.function_names[0])
    g.backwards_breadth_first_visitor(
        visit,
        starting_nodes=[f.get_node_by_name("Identity")])
    self.assertEqual(
        [f.get_node_by_name("Identity"),
         f.get_node_by_name("mul"),
         f.get_node_by_name("function_multiplier")],
        nodes_in_backwards_bfs)

    nodes_in_backwards_bfs = []
    f = g.get_function_graph_by_name(g.function_names[0])
    g.backwards_breadth_first_visitor(
        visit,
        starting_nodes=[f.get_node_by_name("Identity")],
        escape_functions=True)
    self.assertEqual(
        [f.get_node_by_name("Identity"),
         f.get_node_by_name("mul"),
         f.get_node_by_name("function_multiplier"),
         g.get_node_by_name("PartitionedCall"),
         g.get_node_by_name("add"),
         g.get_node_by_name("x"),
         g.get_node_by_name("y")],
        nodes_in_backwards_bfs)

  def test_backwards_breadth_first_visitor_escape_nested_functions(self):
    g = self.build_graph_with_nested_function_call()
    nodes_in_backwards_bfs = []
    def visit(node):
      nodes_in_backwards_bfs.append(node)

    add_node = list(g.nodes_iterator(lambda n:n.name=='add', iterate_functions=True))[0]
    multiplier_function_name = g.get_node_by_name("x").outputs[0].consumers()[0].get_attr('f').name
    multiplier_function_graph = g.get_function_graph_by_name(multiplier_function_name)

    adder_call_op = list(g.nodes_iterator(lambda n:n.op_type=='PartitionedCall' and n.get_attr('f').name == add_node.graph.name, iterate_functions=True))[0]
    multiplier_call_op = list(g.nodes_iterator(lambda n:n.op_type=='PartitionedCall' and n.get_attr('f').name != add_node.graph.name, iterate_functions=True))[0]

    g.backwards_breadth_first_visitor(
        visit,
        starting_nodes=[add_node],
        iterate_functions=True,
        escape_functions=True)
    self.assertEqual(
        [add_node,
         adder_call_op,
         multiplier_call_op,
         g.get_node_by_name("x"),
         g.get_node_by_name("y")],
         nodes_in_backwards_bfs)


if __name__ == "__main__":
  unittest.main()
