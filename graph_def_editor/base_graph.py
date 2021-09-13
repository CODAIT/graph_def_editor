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
"""Base class for Graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
from distutils import dir_util
import os
from six import string_types
import tensorflow.compat.v1 as tf
import sys
if sys.version >= '3':
  from typing import Tuple, Dict, FrozenSet, Iterable, Union, Set, Any

from graph_def_editor import node, util, tensor, variable
import graph_def_editor.visualization.graphviz_wrapper as gvw


__all__ = [
  "BaseGraph",
]

class BaseGraph(object):
  """
  Base class for Graph and FunctionGraph classes.

  Mutable surrogate for a `tf.GraphDef` protocol buffer message.

  Summary of internal data structures:
  * _node_name_to_node: Nodes in the graph, stored as a dictionary. Key is name.
  * _version: Counter that increments every time the graph is modified
  * _collections: Map from collection name to collection contents for all
                  collections
  """

  def __init__(
          self,
          name = None, # type: str
          ):
    """
    Constructor to be called by subclasses only.

    Initializes attributes of this base class.

    Args:
      name: Optional human-readable name for the graph. If not provided,
        the constructor will generate a name.
    """
    # Populate fields of object
    self._name = name  # str
    self._version = 0  # Must happen first; other init code needs self._version
    self._frozen = False  # bool
    self._next_id = 1  # int
    self._node_name_to_node = {}  # Dict[str, node.Node]; key is node name
    self._variable_name_to_variable = {}  # Dict[str, Variable]

  @property
  def name(self):
    """
    Returns human-readable name for this graph. This name may not be unique
    across graphs.
    """
    return self._name

  def __getitem__(self, name):
    # type: (str) -> Union[tensor.Tensor, 'node.Node']
    """
    Convenience method to retrieve a node or tensor of the graph by name

    Args:
      name: Name of the node or tensor to return. Case-sensitive.

    Returns the named item as a `gde.Node` or `gde.Tensor` object. If there
    is a conflict between node and tensor names, node names win.
    """
    if not isinstance(name, string_types):
      raise TypeError("name must be a string; got type {}".format(type(name)))

    if self.contains_node(name):
      return self._node_name_to_node[name]
    elif self.contains_tensor(name):
      return self.get_tensor_by_name(name)
    else:
      raise ValueError("No node or tensor '{}' found in graph".format(name))

  def get_node_by_name(self, name):
    # type: (str) -> node.Node
    """
    Retrieve a node in the graph by name.

    Args:
      name: Name of the node. Case-sensitive.

    Returns the indicated node as a `gde.Node` object.
    """
    if self.contains_node(name):
      return self._node_name_to_node[name]
    else:
      raise ValueError("No node '{}' found in graph".format(name))

  def contains_node(self, name):
    # type: (str) -> bool
    """
    Returns true if the graph has a node by the indicated name. Exact string
    match.
    """
    if not isinstance(name, string_types):
      raise ValueError("Node name argument is not a string, but is of type "
                       "{}".format(type(name)))
    return name in self._node_name_to_node.keys()

  def add_node(self,
               name, # type: str
               op_name, # type: str
               uniquify_name = False, # type: bool
               debug_info = None # type: tf.compat.v1.NodeDef.ExperimentalDebugInfo
               ):
    # type: (...) -> node.Node
    """
    Add a new, empty node to the graph.
    Args:
      name: Name for the new op
      op_name: Name of the type of operation for the node
      uniquify_name: Generate a unique name from this name if the graph
        already has a node with the indicated name. If False, raise an
        exception if the name is in use.
      debug_info: Some internal TensorFlow debug information.
        We just pass it through for safety.

    Returns:
      `MutableNode` wrapper for the new node.

    Raises:
      ValueError if the name is already in use and `uniquify_name` is False
    """
    if uniquify_name:
      name = self.unique_name(name)
    elif self._name_in_use(name):  # and not uniquify_name
      raise ValueError("Graph already contains a node with name '{}' "
                       "(Note that this check is case-insensitive)."
                       .format(name))
    ret = node.Node(self,
                    self._get_next_id(),
                    name=name,
                    op_name=op_name,
                    debug_info=debug_info)
    self._node_name_to_node[name] = ret
    self.increment_version_counter()
    return ret

  def add_node_from_node_def(self,
                             node_def, # type: tf.NodeDef
                             set_inputs = False, # type: bool
                             set_control_inputs = False # type: bool
                             ):
    # type: (...) -> node.Node
    """
    Adds a new node to the graph, populating fields of the node from a
    `tf.NodeDef` protocol buffer.

    Equivalent to calling `add_node()`, then populating the relevant fields
    of the returned MutableNode object.

    Args:
      node_def: Protocol buffer describing parameters of the new node.
      set_inputs: If True, populate the node's inputs list from the list of
        inputs in the `NodeDef`
      set_control_inputs: Also set control inputs. Must be False if
        `set_inputs` is False.

    Returns:
      `MutableNode` wrapper for the new node
    """
    if set_control_inputs and not set_inputs:
      raise ValueError("set_inputs must be True if set_control_inputs is True")
    ret = self.add_node(name=node_def.name,
                        op_name=node_def.op,
                        debug_info=node_def.experimental_debug_info)
    if set_inputs:
      ret.set_inputs_from_strings(node_def.input,
                                  set_control_inputs=set_control_inputs)
    ret.device = node_def.device
    ret.clear_attrs()
    for key in node_def.attr:
      ret.add_attr(key, node_def.attr[key])

    # Don't need to increment version counter; add_node() already did that.
    return ret

  def remove_node_by_name(self, name, check_for_refs = True):
    # type: (str, str) -> None
    """
    Removes the indicated node from this graph and from any collections in
    this graph.

    The caller is responsible for removing all links to the indicated node
    prior to making this call.

    Args:
      name: name of the node to remove
      check_for_refs: Optional. If True, raise an exception if there are any
        other nodes in the graph that reference this node. If False, allow
        removal of nodes with outstanding references to them. In the latter
        case, the caller is responsible for cleaning up the graph afterwards.
    """
    n = self.get_node_by_name(name)
    if check_for_refs:
      for t in n.outputs:
        if len(t.consumers()) > 0:
          raise ValueError("Removing node '{}' would leave dangling "
                           "references from nodes {} to tensor '{}'"
                           "".format(name, [c.name for c in t.consumers()],
                                     t.name))
    # noinspection PyProtectedMember
    n._remove_from_graph()
    del self._node_name_to_node[name]
    self.increment_version_counter()
    # Don't need to update collection info because collection membership is
    # stored in the node.
    # Don't need to update consumers of tensors because that information is
    # calculated dynamically by iterating over nodes.

  def rename_node(self, old_name, new_name):
    # type: (str, str) -> None
    """
    Change the name of a node in the graph.

    Args:
      old_name: Name of an existing node
      new_name: New name for the node in question. Must not currently be in use.
    """
    if self.contains_node(new_name):
      raise ValueError("Graph already has a node under name '{}'".format(
        new_name))
    n = self.get_node_by_name(old_name)
    # noinspection PyProtectedMember
    n._change_name(new_name)
    del self._node_name_to_node[old_name]
    self._node_name_to_node[new_name] = n
    self.increment_version_counter()

  def add_variable(self, name):
    # type: (str) -> variable.Variable
    """
    Adds a new variable to the graph.

    Args:
      name: Name of the variable. Must not already be in use.

    Returns the `gde.Variable` object corresponding to the added variable.
    """
    if name in self._variable_name_to_variable:
      raise ValueError("Variable name '{}' already in use".format(name))
    v = variable.Variable(self)
    v.name = name
    self._variable_name_to_variable[name] = v
    self.increment_version_counter()
    return v

  def add_variable_from_variable_def(self, variable_def,
                                     skip_if_present = False):
    # type: (Any, bool) -> None
    """
    Adds a new variable to the graph and populates the fields of the
    corresponding Variable object according to a protocol buffer message.

    Args:
      variable_def: `tensorflow.core.framework.variable_pb2.VariableDef`
        protobuf object. May be serialized as a `bytes` object.
      skip_if_present: If True, silently skips inserting duplicate variables,
        as long as they don't conflict with existing variables.

    Returns the `gde.Variable` object corresponding to the added variable.
    """
    v = variable.Variable(self)
    v.from_proto(variable_def, allow_duplicates=skip_if_present)
    if v.name not in self._variable_name_to_variable:
      self._variable_name_to_variable[v.name] = v
    return self._variable_name_to_variable[v.name]

  @property
  def variable_names(self):
    return self._variable_name_to_variable.keys()

  def get_variable_by_name(self, name):
    # type: (str) -> variable.Variable
    """
    Fetch a variable by its variable name.

    Args:
      name: Name of a variable in this graph.

    Returns the variable associated with the name. Raises an exception if
    there is no variable with the indicated name.
    """
    return self._variable_name_to_variable[name]

  def _name_in_use(self, name):
    # type: (str) -> bool
    """Check whether a name is in use, using the same collision semantics as
    TensorFlow: Exact lowercase string match.

    Args:
      name: Name of a potential node in the graph.

    Returns True if the indicated name is currently in use, ignoring case.
    """
    return name.lower() in [k.lower() for k in self._node_name_to_node.keys()]

  def unique_name(self, name):
    # type: (str) -> str
    """Emulate the behavior of the method by the same name in `tf.Graph`.

    Does *not* emulate the `name_stack` field of `tf.Graph`.

    Unlike the original method, this version does *not* keep a separate table
    of names currently "in use for the purposes of `unique_name()`", but instead
    refers directly to internal data structures to find names that are truly
    in use.

    Args:
      name: The name for an operation.

    Returns:
      A variant of `name` that has been made unique by appending a key to it
      in the same way that `tf.Graph.unique_name()` would.
    """
    # For the sake of checking for names in use, we treat names as case
    # insensitive (e.g. foo = Foo).
    if not self._name_in_use(name):
      return name

    # Generate a unique version by appending "_1", "_2", etc. until we find
    # an unused name. Note that this approach will behave slightly
    # differently from the original if nodes are deleted.
    i = 1
    new_name = "{}_{}".format(name, i)
    while self._name_in_use(new_name):
      i = i + 1
      new_name = "{}_{}".format(name, i)
    return new_name

  @property
  def node_names(self):
    # type: () -> Iterable[str]
    return self._node_name_to_node.keys()

  @property
  def nodes(self):
    # type: () -> Tuple[node.Node]
    """
    Returns:
      A list of all nodes, both immutable and mutable, present in the graph
      after the edits that this object is buffering.
    """
    return tuple(self._node_name_to_node.values())

  @property
  def tensors(self):
    # type: () -> List[tensor.Tensor]
    """
    Return a list of all the tensors which are input or output of an op in
    the graph.
    """
    ts = []
    for op in self.nodes:
      ts += op.outputs
    return ts

  def contains_tensor(self, tensor_name):
    # type: (str) -> bool
    """
    Returns true if the graph has a tensor by the indicated name. Exact string
    match.

    Args:
      tensor_name: TensorFlow-format name ('node name:input num', or 'node
        name' as shorthand for 'node name:0')

    Raises ValueError if the tensor name is not properly formatted.
    """
    error_msg = "Invalid tensor name '{}': {}"
    node_name, output_ix = self._decode_tensor_name(tensor_name, error_msg)
    if node_name not in self._node_name_to_node:
      return False
    else:
      n = self[node_name]
      if output_ix >= len(n.outputs):
        return False
      else:
        return True

  def get_tensor_by_name(self, tensor_name, error_msg = None):
    # type: (str, str) -> tensor.Tensor
    """
    Retrieve a tensor by human-readable name.

    Args:
      tensor_name: TensorFlow-format name ('node name:input num', or 'node
        name' as shorthand for 'node name:0')
      error_msg: Optional format string for raising errors. Must be able to
        serve as an input to `str.format()` with two arguments: tensor name
        string and reason for failure.

    Returns: gde.Tensor object corresponding to the indicated tensor.

    Raises ValueError if the name is invalid or references a tensor that does
    not exist.
    """
    if error_msg is None:
      error_msg = "Invalid tensor name '{}': {}"
    node_name, output_ix = self._decode_tensor_name(tensor_name, error_msg)
    if node_name not in self._node_name_to_node:
      raise ValueError(error_msg.format(
        tensor_name, "Node name '{}' not found in graph.".format(node_name)
      ))
    n = self[node_name]
    if output_ix >= len(n.outputs):
      raise ValueError(error_msg.format(
        tensor_name, "Requested output {}, but node '{}' has {} "
                     "outputs.".format(output_ix, node_name, len(n.outputs))
      ))
    return n.output(output_ix)

  @property
  def version(self):
    # type: () -> int
    """
    Returns a counter that goes up every time this graph is changed.
    """
    return self._version

  @property
  def frozen(self):
    # type: () -> bool
    """
    True if the graph is configured to raise an exception on any structural
    modification.
    """
    return self._frozen

  @frozen.setter
  def frozen(self, value):
    # type: (bool) -> None
    self._frozen = value

  def increment_version_counter(self):
    """
    Mark the structure of this graph as "changed" and invalidate any cached
    information about the edges of the graph.
    """
    if self.frozen:
      raise RuntimeError("Detected a change to a frozen graph")
    self._version += 1

  def visualize(
      self,
      format=None,
      depth=1,
      style=True,
      name_regex="",
      negative_name_regex="",
      add_digraph_func=None,
      add_digraph_node_func=None,
      add_digraph_edge_func=None):
    """Return GraphViz Digraph rendering of the current graph.

    Args:
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

    Returns:
      graphviz.dot.Digraph object with visual representtion for the current
        graph.
    """
    return gvw.visualize(
        self,
        format=format,
        depth=depth,
        name=self.name,
        style=style,
        name_regex=name_regex,
        negative_name_regex=negative_name_regex,
        add_digraph_func=add_digraph_func,
        add_digraph_node_func=add_digraph_node_func,
        add_digraph_edge_func=add_digraph_edge_func)

  def _get_next_id(self):
    # type: () -> int
    """Generates and returns a unique integer ID *within this graph*."""
    ret = self._next_id
    self._next_id = ret + 1
    return ret

  def _decode_tensor_name(self, tensor_name, error_msg):
    # type: (str, str) -> Tuple[str, int]
    """
    Args:
      tensor_name: TensorFlow-format name ('node name:input num', or 'node
        name' as shorthand for 'node name:0')
      error_msg: Format string for raising errors. Must be able to
        serve as an input to `str.format()` with two arguments: tensor name
        string and reason for failure.

    Returns: (node name, output index) tuple identifying the tensor

    Raises ValueError if the name is not properly formatted
    """
    if ":" in tensor_name:
      node_name, output_ix_str = tensor_name.split(":")
      if not output_ix_str.isdigit():
        raise ValueError(error_msg.format(
          tensor_name, "Invalid output index string '{}'.".format(output_ix_str)
        ))
      output_ix = int(output_ix_str)
    else:
      node_name = tensor_name
      output_ix = 0

    return node_name, output_ix

