# Copyright 2018 IBM. All Rights Reserved.
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
"""Objects for representing entire graphs undergoing rewrite operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Tuple, Dict, FrozenSet, Iterable

from gde import node
from gde import util
from gde import variable

__all__ = [
  "Graph",
]

# Special attribute in which TensorFlow stores frame names for while loops (
# see node_to_frame_name() for more information
_FRAME_NAME_ATTR = "frame_name"


class Graph(object):
  """
  Mutable surrogate for a `tf.GraphDef` protocol buffer message

  Also stores collection information that would be serialized in MetaGraphDef

  Summary of internal data structures:
  * _node_name_to_node: Nodes in the graph, stored as a dictionary. Key is name.
  * _version: Counter that increments every time the graph is modified
  * _collections: Map from collection name to collection contents for all
                  collections
  """

  def __init__(self, g: tf.GraphDef = None, collections:
               Iterable[tf.MetaGraphDef.CollectionDefEntry] = None):
    """
    Wrap a tf.GraphDef protocol buffer in a Graph object.

    Args:
      g: a tf.Graph or tf.GraphDef protobuf that represents a
        TensorFlow graph. If set to None, generate an empty
        tf.GraphDef
      collections: Optional iterable of tf.MetaGraphDef.CollectionDefEntry 
        objects containing information about collections in the graph.
        Note that this constructor will pull collection info out of `g` if
        it is a `tf.Graph` and `collections` is `None`.
    """
    if g is None:
      graph_def = tf.GraphDef()
    elif isinstance(g, tf.GraphDef):
      graph_def = g
    elif isinstance(g, tf.Graph):
      graph_def = g.as_graph_def()
      if collections is None:
        collections = _make_collection_defs(g)
    else:
      raise TypeError("Graph is of type {}. Expected a tf.Graph or GraphDef "
                      "proto".format(type(g)))
    self._version = 0  # Must happen first; other init code needs self._version
    self._frozen = False
    self._graph_def = graph_def
    self._next_id = 1
    output_map = _decode_graph(graph_def)
    self._node_name_to_node = {}  # Dict[str, node.Node]; key is node name
    self._node_to_frame_names = None
    self._frame_name_to_nodes = None
    self._head_name_to_coloc_group = None  # Dict[str, FrozenList[str]]
    self._variable_name_to_variable = {}  # Dict[str, Variable]

    # Load nodes in three passes because the g may contain cycles.
    for node_def in graph_def.node:
      self.add_node_from_node_def(node_def, set_inputs=False)
    for node_def in graph_def.node:
        self[node_def.name].set_outputs_from_pairs(output_map[node_def.name])
    for node_def in graph_def.node:
      self[node_def.name].set_inputs_from_strings(node_def.input,
                                                  set_control_inputs=True)

    self._collections = {}
    if collections is not None:
      for c in collections:
        self.add_collection_from_collection_def(c)

  def add_node_from_node_def(self, node_def: tf.NodeDef,
                             set_inputs: bool = False) -> node.Node:
    """
    Unpack a `tf.NodeDef` protobuf into a mutable `Node` object.'

    Does NOT set the outputs of the node.

    Args:
      g: Graph in which the node will be created
      node_def: Fully-populated NodeDef proto; all fields, including inputs,
        will be used.
      set_inputs: Optional. If True, also populate the data and control inputs of
        the returned Node. This operation will only work if the targets of those
        inputs are already present in the graph.
    """
    ret = self.add_node(name=node_def.name, op_name=node_def.op)
    ret.device = node_def.device
    for key in node_def.attr:
      ret.add_attr(key, util.attr_value_to_python_type(node_def.attr[key]))
    if set_inputs:
      ret.set_inputs_from_strings(node_def.input, set_control_inputs=True)
    return ret

  def add_collection_from_collection_def(self, collection_def:
                                         tf.MetaGraphDef.CollectionDefEntry):
    """
    Unpack a `tf.MetaGraphDef.CollectionDefEntry` of serialized variables 
    into a collection of variables in this graph. The collection must not exist. 
    Variables that do not already exist will be created.
    """
    collection_name = collection_def.key
    for serialized_var in collection_def.value.bytes_list.value:
      var = self.add_variable_from_variable_def(serialized_var,
                                                skip_if_present=True)
      var.add_to_collection(collection_name)

  def __getitem__(self, name: str) -> node.Node:
    """
    Retrieve a node of the graph by name

    Args:
      name: Name of the node to return
    """
    if not isinstance(name, str):
      raise TypeError("name must be a string; got type {}".format(type(name)))

    # Search the diffs first, then go back to the original immutable graph
    if self.contains_node(name):
      return self._node_name_to_node[name]
    else:
      raise ValueError("No node '{}' found in graph".format(name))

  def contains_node(self, name: str) -> bool:
    """
    Returns true if the graph has a node by the indicated name. Exact string
    match.
    """
    return name in self._node_name_to_node.keys()

  def add_node(self, name: str, op_name: str, uniquify_name: bool = False) -> \
          node.Node:
    """
    Add a new, empty node to the graph.
    Args:
      name: Name for the new op
      op_name: Name of the type of operation for the node
      uniquify_name: Generate a unique name from this name if the graph
        already has a node with the indicated name. If False, raise an
        exception if the name is in use.

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
    ret = node.Node(self, self._get_next_id(), name=name, op_name=op_name)
    self._node_name_to_node[name] = ret
    self.increment_version_counter()
    return ret

  def add_node_from_node_def(self, node_def: tf.NodeDef,
                             set_inputs: bool = False,
                             set_control_inputs: bool = False) -> node.Node:
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
    ret = self.add_node(node_def.name, node_def.op)
    if set_inputs:
      ret.set_inputs_from_strings(node_def.input,
                                  set_control_inputs=set_control_inputs)
    ret.device = node_def.device
    ret.clear_attrs()
    for key in node_def.attr:
      ret.add_attr(key, node_def.attr[key])

    # Don't need to increment version counter; add_node() already did that.
    return ret

  def add_variable(self, name: str):
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
    return v

  def add_variable_from_variable_def(self, variable_def,
                                     skip_if_present: bool = False):
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

  def name_to_variable(self, name: str) -> variable.Variable:
    """
    Fetch a variable by its variable name.

    Args:
      name: Name of a variable in this graph.

    Returns the variable associated with the name. Raises an exception if
    there is no variable with the indicated name.
    """
    return self._variable_name_to_variable[name]

  def _name_in_use(self, name: str) -> bool:
    """Check whether a name is in use, using the same collision semantics as
    TensorFlow: Exact lowercase string match.

    Args:
      name: Name of a potential node in the graph.

    Returns True if the indicated name is currently in use, ignoring case.
    """
    return name.lower() in [k.lower() for k in self._node_name_to_node.keys()]

  def unique_name(self, name: str):
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
  def node_names(self) -> Iterable[node.Node]:
    return self._node_name_to_node.keys()

  @property
  def nodes(self) -> Tuple[node.Node]:
    """
    Returns:
      A list of all nodes, both immutable and mutable, present in the graph
      after the edits that this object is buffering.
    """
    return tuple(self._node_name_to_node.values())

  @property
  def tensors(self):
    """
    Return a list of all the tensors which are input or output of an op in
    the graph.
    """
    ts = []
    for op in self.nodes:
      ts += op.outputs
    return ts

  def get_tensor_by_name(self, tensor_name: str, error_msg: str = None):
    """
    Retrieve a tensor by human-readable name.

    Args:
      tensor_name: TensorFlow-format name ('node name:input num')
      error_msg: Optional format string for raising errors. Must be able to
        serve as an input to `str.format()` with two arguments: tensor name
        string and reason for failure.

    Returns: gde.Tensor object corresponding to the indicated tensor.

    Raises ValueError if the name is invalid or references a tensor that does
    not exist.
    """
    if error_msg is None:
      error_msg = "Invalid tensor name '{}': {}"

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
    return n.output(int(output_ix_str))

  def to_graph_def(self):
    """
    Returns the `tf.GraphDef` serialization of this graph in its current
    form.
    """
    ret = tf.GraphDef()
    for op in self.nodes:
      op.to_node_def(ret.node.add())
    return ret

  @property
  def version(self):
    """
    Returns a counter that goes up every time this graph is changed.
    """
    return self._version

  @property
  def frozen(self):
    """
    True if the graph is configured to raise an exception on any structural
    modification.
    """
    return self._frozen

  @frozen.setter
  def frozen(self, value):
    self._frozen = value

  def increment_version_counter(self):
    """
    Mark the structure of this graph as "changed" and invalidate any cached
    information about the edges of the graph.
    """
    if self.frozen:
      raise RuntimeError("Detected a change to a frozen graph")
    self._version += 1
    self._node_to_frame_names = None
    self._frame_name_to_nodes = None
    self._head_name_to_coloc_group = None

  def get_collection(self, name: str):
    """Fetch the contents of a collection, similarly to the method in
    `tf.Graph` by the same name.

    Args:
      name: Name of collection to fetch

    Returns:
      The values in the collection. Currently any type is allowed in these
      values, following the conventions of the TensorFlow APIs.
    """
    return self._collections[name]

  def get_all_collection_keys(self):
    """Returns the keys associated with all collections stored in this object"""
    return self._collections.keys()

  def _get_next_id(self) -> int:
    """Generates and returns a unique integer ID *within this graph*."""
    ret = self._next_id
    self._next_id = ret + 1
    return ret

  def node_to_frame_names(self, n: node.Node) -> Tuple[str]:
    """
    Generates (or uses a cached copy of) a map from graph node to the name of
    the associated control flow frame(s).

    *A word about frames*

    The while loop construct in TensorFlow is built on the concept of a
    "frame", which is kind of like a stack frame, except that the "stack" is
    a directed acyclic graph. When creating a while loop, TensorFlow inserts
    special passthrough placeholder ops "Enter" and "Exit" ("RefEnter" and
    "RefExit" when passing through references to tensors) before and after
    the cycle in the operator graph that makes up the loop. These ops do not
    perform any work themselves, but the TensorFlow scheduler recognizes them
    as special "control instructions", creating and destroying frames when
    executing the "Enter" and "Exit" ops.

    Rather than maintain a data structure that represents the current
    DAG of frames, TensorFlow's scheduler performs a static analysis of the
    graph, associating each op with a chain of "parent" control ops that
    comprise the "stack" when the op executes. Each parent op is associated
    with a unique string called the "frame name", which, by convention,
    is stored in the "frame_name" attribute of every "Enter" op leading into
    a given while loop. At runtime, the actual names of frame instances
    are generated by appending iteration numbers onto these frame name strings.

    It is important that the value of the "frame_name" attribute be the same for
    every "Enter" op leading into the while loop's body; and different for
    every while loop in the graph. If these constraints are not met,
    execution of the graph will go awry in nasty, messy ways. So be sure to
    maintain proper frame info when rewriting GraphDefs.

    It is also important that the set of "parent" nodes for a given node be
    unambiguous. The static analysis in the TensorFlow scheduler is only
    correct for the types of control flow that the high-level Python APIs can
    generate. Be very careful when implementing rewrites that could
    lead to graph topologies (for example, sharing a loop body among multiple
    while loops) that are impossible to build with TensorFlow's Python APIs.

    Args:
      n: Node in this graph

    Returns:
      Dictionary mapping from nodes in this graph to the names of one or
      more nested frames that will be active when reaching this node.
      The returned value is only valid until this graph is modified, either
      by modifying the link structure of the graph or by changing the
      "frame_name" attribute of an Enter node. Nodes that are not nested
      inside any while loops are mapped to None.
    """
    if self._node_to_frame_names is None:
      self._generate_node_to_frame_name()
    return self._node_to_frame_names[n]

  def frame_name_to_nodes(self, frame_name: str) -> Tuple[node.Node]:
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

  def get_frame_names(self) -> Tuple[str]:
    """
    Returns:
      Tuple of all the unique names of frames that occur in this graph.
    """
    if self._node_to_frame_names is None:
      self._generate_node_to_frame_name()
    return self._frame_name_to_nodes.keys()

  def _generate_node_to_frame_name(self):
    """
    Regenerate the tables behind the node_to_frame_name and
    frame_name_to_node properties. Performs a breadth-first traversal of the
    graph, duplicating the logic in the function
    ExecutorImpl::BuildControlFlowInfo() in
    tensorflow/core/common_runtime/executor.cc
    """
    new_node_to_frame_names = {}
    new_frame_name_to_nodes = {}

    # Use a Python list as a node queue for the breadth-first search.
    # Start with all of the nodes in the graph that have no inputs.
    # The maintainers of the TensorFlow scheduler like to call these nodes
    # "root nodes".
    queue = [n for n in self.nodes if 0 == len(n.inputs)]
    visited = set(queue)  # Invariant: visited == all nodes enqueued

    # Use a second list to maintain a stack of frame names, a la the original
    # code in executor.cc.
    # We use None to represent the root frame.
    frame_name_stack = [None]
    frame_name_tuple = tuple()  # Immutable version for table

    # Breadth-first search; same algorithm as in the C++ code, except that
    # instead of keeping a stack of parent ops we keep a stack of frame names.
    while len(queue) > 0:
      # Update data structures that track the stack
      cur_node = queue.pop(0)
      if cur_node.op_type in ["Enter", "RefEnter"]:
        # Entering a while loop. Push a frame name onto the virtual stack
        if _FRAME_NAME_ATTR not in cur_node.get_attr_keys():
          raise ValueError("Node {} is of op type {} but does not have a "
                           "value for its {}"
                           " attribute".format(cur_node.name,
                                               cur_node.op_type,
                                               _FRAME_NAME_ATTR))
        frame_name_stack.append(cur_node.get_attr(_FRAME_NAME_ATTR))
        frame_name_tuple = tuple(frame_name_stack)
      elif cur_node.op_type in ["Exit", "RefExit"]:
        frame_name_stack.pop(-1)
        frame_name_tuple = tuple(frame_name_stack)
      # Update tables
      new_node_to_frame_names[cur_node] = frame_name_tuple
      for frame_name in frame_name_stack:
        new_frame_name_to_nodes.setdefault(frame_name, []).append(cur_node)
      # Prepare for next stage of search
      for out_tensor in cur_node.outputs:
        for out_node in out_tensor.consumers():
          if out_node not in visited:
            queue.append(out_node)
            visited.add(out_node)

    self._node_to_frame_names = new_node_to_frame_names
    # Reverse mapping was built a dict of lists to avoid O(n^2) behavior.
    # Convert to dict of tuples.
    self._frame_name_to_nodes = {
      k: tuple(v) for k, v in new_frame_name_to_nodes.items()
    }

  @property
  def colocation_groups(self) -> Dict[str, FrozenSet[node.Node]]:
    """
    Generate a table of all groups of nodes that must be on the same device
    according to colocation constrains in the underlying NodeDefs.

    Returns:
      A dictionary with one entry per group. Key is the name of the
      "master" node in the group; value is a set of nodes.
      The returned value will become invalid if colocation group info or
      graph topology is updated.
    """
    if self._head_name_to_coloc_group is None:
      # Cached table has been invalidated. Regenerate it.
      head_name_to_coloc_group = {}  # Dict[str, Set[str]]
      for n in self.nodes:
        for head_name in n.colocation_groups:
          head_name_to_coloc_group.setdefault(head_name, set()).add(n)
      self._head_name_to_coloc_group = {
        k: frozenset(v) for k, v in head_name_to_coloc_group.items() }
    return self._head_name_to_coloc_group


################################################################################
# Stuff below this line is private to this file.


def _decode_graph(graph_def):
  """
  Use public TensorFlow APIs to decode the important information that is not
  explicitly stored in the GraphDef proto, but which must be inferred from the
  GraphDef in conjunction with additional data structures that TensorFlow
  generally keeps to itself.

  Args:
    graph_def: tf.GraphDef protobuf that represents a TensorFlow graph.
      This graph must be runnable on the current version of TensorFlow;
      otherwise some of the type inference operations that this function
      performs will fail.

  Returns:
    A map from node name to a list of (type, shape) pairs that describe
    in turn each of the outputs of said node.
  """
  # The information in a NodeDef is not sufficient to determine output type
  # information. For that kind of type inference, you need access to the
  # corresponding OpDef protos. Unfortunately there is not a public API that
  # allows for OpDef lookup. So instead we instantiate the graph that
  # graph_def describes. This approach makes things easier, but there will be
  # a reduction in forwards compatibility, because import_graph_def() does a
  # lot of sanity checks that aren't necessary when rewriting a graph_def.
  temp_graph = tf.Graph()
  with temp_graph.as_default():
    tf.import_graph_def(graph_def, name="")
  output_map = {op.name: [(t.dtype, t.shape) for t in op.outputs]
                for op in temp_graph.get_operations()}
  return output_map


def _make_collection_defs(tf_g: tf.Graph) -> Iterable[
  tf.MetaGraphDef.CollectionDefEntry]:
  """
  Convenience function to serialize all the collections in a TensorFlow graph.

  **NOTE:** Currently this function only captures collections of variables.

  Args:
    tf_g: TensorFlow graph from which to harvest collections

  Returns a list of `tf.MetaGraphDef.CollectionDefEntry` protobuf containing
  the serialized
  contents of the collections.
  """
  ret = []
  for collection_name in tf_g.collections:
    if type(collection_name) is not str:
      print("Skipping non-string collection name {}".format(collection_name))
      continue
    collection_items = tf_g.get_collection(collection_name)
    collection_proto = tf.MetaGraphDef.CollectionDefEntry()
    collection_proto.key = collection_name
    for item in collection_items:
      if isinstance(item, tf.Variable):
        # Ask TensorFlow to generate the protobuf version of this variable
        var_proto = item.to_proto()

        # TensorFlow stores variables as binary serialized objects for some
        # reason.
        collection_proto.value.bytes_list.value.append(
          var_proto.SerializeToString())
      elif type(item).__name__ == "WhileContext":
        # TODO(frreiss): Should we serialize WhileContexts?
        print("Skipping collection {} -- is WhileContext.".format(
          collection_name))
      elif type(item).__name__ == "CondContext":
        # TODO(frreiss): Should we serialize CondContexts?
        print("Skipping collection {} -- is CondContext.".format(
          collection_name))
      else:
        raise NotImplementedError("Can't serialize item '{}' in collection "
                                  "'{}' because it is a "
                                  "'{}'.".format(item, collection_name,
                                                 type(item).__name__))

    ret.append(collection_proto)
  return ret

