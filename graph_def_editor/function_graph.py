# Copyright 2021 Google. All Rights Reserved.
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
"""Objects for representing function graphs undergoing rewrite operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import datetime
from distutils import dir_util
import os
from six import string_types
import tensorflow.compat.v1 as tf
import sys
if sys.version >= "3":
  from typing import Tuple, Dict, List, FrozenSet, Iterable, Union, Set, Any

from graph_def_editor import base_graph, node, util, tensor, variable

# TODO: Move this protobuf into this project so we don't depend on
#  tf.core.framework
from tensorflow.core.framework import function_pb2, op_def_pb2
from tensorflow.python.framework import function_def_to_graph


__all__ = [
  "FunctionGraph",
]

# Special attribute in which TensorFlow stores frame names for while loops (
# see node_to_frame_name() for more information
_INPUT_DUMMY_OP_NAME = "__input__"


class FunctionGraph(base_graph.BaseGraph):
  """Wrapper class for TensorFlow function graphs."""

  def __init__(
      self,
      name=None,  # type: str
      parent_tf_graph=None,  # type: tf.Graph
      parent_graph=None  # type: gde.Graph
  ):
    """Wrap a tf.GraphDef protocol buffer in a FunctionGraph object.

    Args:
      g: a tf.Graph or tf.GraphDef protobuf that represents a
        TensorFlow graph. If set to None, generate an empty
        tf.GraphDef
      name: Optional human-readable name for the graph. If not provided,
        the constructor will generate a name.
    """
    super(FunctionGraph, self).__init__(name)
    (self._func_graph, self._func_graph_def) = \
        _get_func_graph_for_name(parent_tf_graph, name)
    output_map = _decode_graph(name, self._func_graph)
    output_map_pairs = {}
    for op_name, tuples in output_map.items():
      output_map_pairs[op_name] = \
          [(dtype, shape) for (dtype, shape, _) in tuples]

    # Populate fields of object
    self._node_to_frame_names = None
    self._frame_name_to_nodes = None
    self._head_name_to_coloc_group = None  # Dict[str, FrozenList[str]]
    self._variable_name_to_variable = {}  # Dict[str, Variable]
    self._collection_name_to_type = None  # Dict[str, str], generated on demand
    self._input_nodes = []
    self._output_nodes = []
    self._parent_graph = parent_graph

    for input_arg in self._func_graph_def.signature.input_arg:
      self._input_nodes.append(
          self.add_node(input_arg.name, _INPUT_DUMMY_OP_NAME))
      self[input_arg.name].set_outputs_from_pairs(
          output_map_pairs[input_arg.name])

    # Load nodes in three passes because the g may contain cycles.
    for node_def in self._func_graph_def.node_def:
      self.add_node_from_node_def(node_def, set_inputs=False)
    for node_def in self._func_graph_def.node_def:
      self[node_def.name].set_outputs_from_pairs(
          output_map_pairs[node_def.name])
    for node_def in self._func_graph_def.node_def:
      try:
        self[node_def.name].set_inputs_from_strings(
            node_def.input,
            set_control_inputs=True,
            output_map=output_map)
      except Exception as ex:
        print("can't set inputs for node: {}; reason: {}".format(
            node_def.name, ex))

    for output_tensor in self._func_graph.outputs:
      self._output_nodes.append(self.get_node_by_name(output_tensor.op.name))

  @property
  def input_nodes(self):
    return self._input_nodes

  @property
  def output_nodes(self):
    return self._output_nodes

  @property
  def parent_graph(self):
    return self._parent_graph

  def get_func_graph_for_name(self, graph, func_name):
    """Returns the FuncGraph associated to the given func_name if possible."""
    outer_graph = graph
    while graph is not None:
      # pylint: disable=protected-access
      func = graph._get_function(str(func_name))
      if func is not None:
        if hasattr(func, "graph"):
          return func.graph
        # `outer_graph` may not be the same as `ops.get_default_graph()` e.g.
        # in the case of nested if ops or when the gradient is being computed
        # from inside a Defun. We build the `func_graph` with `outer_graph`
        # as its outer graph.
        with outer_graph.as_default():
          # This is a _DefinedFunction.
          func_graph = (
              function_def_to_graph.function_def_to_graph(func.definition))
        if func_graph is not None:
          return func_graph
      if hasattr(graph, "outer_graph"):
        graph = graph.outer_graph
      else:
        raise ValueError(
            "Function {} does not exist in the graph.".format(func_name))

  def to_function_graph_def(self, add_shapes=True):
    # type: (bool) -> function_pb2.FunctionDef
    """
    Args:
      add_shapes: If True, add the special "_output_shapes" attribute with
        output shape information from this Node's output metadata.

    Returns the `function_pb2.FunctionDef` serialization of this function's
    graph in its current form.
    """
    ret = function_pb2.FunctionDef()
    ret.CopyFrom(self._func_graph_def)
    # Leave signature as is, but replace all node_defs
    del ret.node_def[:]
    ret.signature.CopyFrom(self._func_graph_def.signature)

    input_args = [input_arg.name for input_arg in ret.signature.input_arg]

    for op in self.nodes:
      if op.op_type == _INPUT_DUMMY_OP_NAME:
        continue

      node_def = ret.node_def.add()
      op.to_node_def(node_def, add_shapes)
      unique_input_counter = Counter()

      for i in range(len(op.inputs)):
        (input_tensor_name, global_input_index_str) = (
            op.inputs[i].name.split(":"))

        global_input_index = int(global_input_index_str)
        if input_tensor_name in input_args:
          # don't add index for function args
          node_def.input[i] = input_tensor_name
        else:
          input_op_output_args, input_op_output_has_number_attr = (
              self._get_op_def_denormalized_outputs(op.inputs[i].op))
          if (len(input_op_output_args) == 1 and
              input_op_output_args[0].type_list_attr):
            node_def.input[i] = (
                input_tensor_name + ":" + input_op_output_args[0].name + ":" +
                str(global_input_index))
          else:
            input_name = (
                input_tensor_name + ":" +
                input_op_output_args[global_input_index].name)
            node_def.input[i] = (
                input_name + ":" + str(unique_input_counter[input_name]))
            if input_op_output_has_number_attr:
              # only uniquify input args with var length,
              # otherwise it should be 0
              unique_input_counter[input_name] += 1
    return ret

  def to_tf_function_graph(self):
    # type: () -> tf.Graph
    """
    Converts this graph into a new TensorFlow `Graph`. Also takes care of
    variables.
    Note that function_def_to_graph.function_def_to_graph won't work if
    function calls into other functions.

    Returns a fresh `tf.Graph` containing all the nodes and variables that
    this object represents.
    """
    return function_def_to_graph.function_def_to_graph(
        self.to_function_graph_def())

  def increment_version_counter(self):
    """
    Mark the structure of this graph as "changed" and invalidate any cached
    information about the edges of the graph.
    """
    super(FunctionGraph, self).increment_version_counter()
    self._node_to_frame_names = None
    self._frame_name_to_nodes = None
    self._head_name_to_coloc_group = None
    self._collection_name_to_type = None

  def frame_name_to_nodes(self, frame_name):
    # type: (str) -> Tuple[node.Node]
    """
    Performs the inverse mapping of node_to_frame_name().

    Args:
      frame_name: Name of a control flow frame in the graph

    Returns:
      All nodes that are tagged with the indicated frame, either as an
      innermost frame or as a containing frame.
    """
    if self._node_to_frame_names is None:
      self._generate_node_to_frame_name()
    return self._frame_name_to_nodes[frame_name]

  def get_frame_names(self):
    # type: () -> Tuple[str]
    """
    Returns:
      Tuple of all the unique names of frames that occur in this graph.
    """
    if self._node_to_frame_names is None:
      self._generate_node_to_frame_name()
    return self._frame_name_to_nodes.keys()

  def _get_op_def_denormalized_outputs(self, op):
    # type: (Node) -> (List[op_def_pb2.OpDef.ArgDef], bool)
    # pylint: disable=protected-access
    op_def = self._func_graph._get_op_def(op.op_type)
    output_args = []

    input_op_output_has_number_attr = False
    for output_arg in op_def.output_arg:
      if output_arg.number_attr:
        l = op.get_attr(output_arg.number_attr)
        input_op_output_has_number_attr = True
        for _ in range(l):
          output_args.append(op_def_pb2.OpDef.ArgDef(name=output_arg.name,
                                                     type=output_arg.type))
      else:
        output_args.append(output_arg)

    return (output_args, input_op_output_has_number_attr)

  def _visualize_node(
      self,
      gde_node,
      format=None,
      depth=1,
      style=True,
      name_regex="",
      negative_name_regex="",
      add_digraph_func=None,
      add_digraph_node_func=None,
      add_digraph_edge_func=None,
      depth_before=1,
      depth_after=2):
    """Return GraphViz Digraph rendering of the current and adjacent nodes.

    Args:
      gde_node: a node to visualize.
      format: GraphViz display format. In addition to that it supports
        jupyter_svg, and jupyter_interactive modes.
      depth: the maximum depth of the graph to display.
      style: whether to apply default styles.
      name_regex: only diplay nodes that have name matching this regex.
      negative_name_regex: only diplay nodes that have name not matching this
        regex.
      add_digraph_func: custom override for function for adding subraphs
        to the resulting Digraph object.
      add_digraph_node_func: custom override for function for adding nodes
        (vertices) to the resulting Digraph object.
      add_digraph_edge_func: custom override for function for adding edges
        to the resulting Digraph object.
      depth_before: number of adjacent nodes to show before the current one.
      depth_after: number of adjacent nodes to show after the current one.

    Returns:
      graphviz.dot.Digraph object with visual representtion for the current
        graph.
    """

    # pylint: disable=protected-access
    return self._parent_graph._visualize_node(
        gde_node=gde_node,
        format=format,
        depth=depth,
        style=style,
        name_regex=name_regex,
        negative_name_regex=negative_name_regex,
        add_digraph_func=add_digraph_func,
        add_digraph_node_func=add_digraph_node_func,
        add_digraph_edge_func=add_digraph_edge_func,
        depth_before=depth_before,
        depth_after=depth_after)


################################################################################
# Stuff below this line is private to this file.


def _get_func_graph_for_name(graph, func_name):
  """Returns the FuncGraph and FuncDef associated to the given func_name."""
  outer_graph = graph
  while graph is not None:
    # pylint: disable=protected-access
    func = graph._get_function(str(func_name))
    if func is not None:
      if hasattr(func, "graph"):
        return (func.graph, func.definition)
      # `outer_graph` may not be the same as `ops.get_default_graph()` e.g.
      # in the case of nested if ops or when the gradient is being computed
      # from inside a Defun. We build the `func_graph` with `outer_graph` as its
      # outer graph.
      with outer_graph.as_default():
        # This is a _DefinedFunction.
        func_graph = (
            function_def_to_graph.function_def_to_graph(func.definition))
      if func_graph is not None:
        return (func_graph, func.definition)
    if hasattr(graph, "outer_graph"):
      graph = graph.outer_graph
    else:
      raise ValueError(
          "Function {} does not exist in the graph.".format(func_name))


def _decode_graph(name, func_graph):
  # type: (str, tf.Graph) -> Dict[str, List[Tuple[tf.DType, tf.TensorShape, str]]]
  """
  Use public TensorFlow APIs to decode the important information that is not
  explicitly stored in the GraphDef proto, but which must be inferred from the
  GraphDef in conjunction with additional data structures that TensorFlow
  generally keeps to itself.

  Args:
    name: function name.
    func_graph: tf.GraphDef protobuf that represents a function graph.

  Returns:
    A map from node name to a list of (type, shape, output_arg_name) tuples
    that describes in turn each of the outputs of said node.
  """
  # The information in a NodeDef is not sufficient to determine output type
  # information. For that kind of type inference, you need access to the
  # corresponding OpDef protos. Unfortunately there is not a public API that
  # allows for OpDef lookup. So instead we instantiate the graph that
  # graph_def describes. This approach makes things easier, but there will be
  # a reduction in forwards compatibility, because import_graph_def() does a
  # lot of sanity checks that aren't necessary when rewriting a graph_def.
  output_map = {}
  for op in func_graph.get_operations():
    # pylint: disable=protected-access
    op_def = func_graph._get_op_def(op.type)
    output_idx = 0
    output_map[op.name] = []
    for output_arg_idx in range(len(op_def.output_arg)):
      output_arg = op_def.output_arg[output_arg_idx]
      output = op.outputs[output_idx]
      if output_arg.type_list_attr:
        output_map[op.name] = [(
            output.dtype, output.shape, op_def.output_arg[0].name)
                               for output in op.outputs]
        break
      elif output_arg.number_attr:
        output_len = op.node_def.attr[output_arg.number_attr].i
        for _ in range(output_len):
          output = op.outputs[output_idx]
          output_map[op.name].append(
              (output.dtype, output.shape, output_arg.name))
          output_idx += 1
      else:
        output_map[op.name].append(
            (output.dtype, output.shape, output_arg.name))
        output_idx += 1
  return output_map

