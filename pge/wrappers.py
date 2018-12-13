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
"""Wrapper classes for the protobufs that comprise TensorFlow's stable public 
API to graph internals."""

import tensorflow as tf
from typing import Tuple, List

__all__ = [
    "Node",
    "Graph",
]


class Graph(object):
  """
  Mutable wrapper for tf.GraphDef.

  Stores a reference to the original immutable tf.GraphDef and information about
  changes made to the graph.

  Summary of internal data structures:
  * _immutable_nodes: The original immutable GraphDef protobuf
  * _deleted_nodes: Tombstones for nodes removed from the original graph,
                    stored as a set of strings. String == name of removed node
  * _added_nodes: New nodes added to the graph, stored as a dictionary. Key
                  is name.
  * _version: Counter that increments every time the graph is modified
  """

  def __init__(self, graph: tf.GraphDef):
    """
    Wrap a tf.GraphDef protocol buffer in a Graph object.

    Args:
      graph: a tf.Graph or tf.GraphDef protobuf that represents a
        TensorFlow operator graph.
    """
    if isinstance(graph, tf.GraphDef):
      graph_def = graph
    elif isinstance(graph, tf.Graph):
      graph_def = graph.as_graph_def()
    else:
      raise TypeError("Graph is of type {}. Expected a tf.Graph or GraphDef "
                      "proto".format(type(graph)))
    self._graph_def = graph_def
    output_map = _decode_graph(graph_def)
    self._immutable_nodes = [ImmutableNode(self, n, output_map[n.name]) for n in
                             graph_def.node]
    self._deleted_nodes = set()
    self._added_nodes = {}
    self._version = 0

  def __getitem__(self, name: str) -> 'Node':
    """
    Retrieve a node of the graph by name

    Args:
      name: Name of the node to return
    """
    if not isinstance(name, str):
      raise TypeError("name must be a string; got type {}".format(type(name)))

    # Search the diffs first, then go back to the original immutable graph
    if name in self._added_nodes.keys():
      return self._added_nodes[name]
    if name in self._deleted_nodes:
      raise ValueError("Node '{}' has been deleted from the graph".format(name))

    ixs = [i for i in range(len(self._immutable_nodes))
           if self._immutable_nodes[i].name == name]
    if 0 == len(ixs):
      raise ValueError("No node '{}' found in graph".format(name))
    elif len(ixs) > 1:
      raise ValueError("Found {} nodes with name '{}' in graph".format(
        len(ixs), name))
    else:
      return self._immutable_nodes[ixs[0]]

  def add_node(self, name: str, op: str):
    """
    Add a new node to the graph.
    Args:
      name: Unique name for the new op
      op: Name of the type of operation for the node

    Returns:
      MutableNode wrapper for the new node
    """
    if name in self._added_nodes.keys() or name in [d.name for d in
                                                    self._graph_def.node]:
      raise ValueError("Graph already contains a node with name '{}'".format(
        name))
    ret = MutableNode(self, name, op)
    self._added_nodes[name] = ret
    self._increment_version_counter()
    return ret

  @property
  def nodes(self) -> Tuple['Node']:
    """
    Returns:
      A list of all nodes, both immutable and mutable, present in the graph
      after the edits that this object is buffering.
    """
    return tuple([n for n in self._immutable_nodes
                  if n.name not in self._deleted_nodes]
                 + list(self._added_nodes))

  @property
  def tensors(self):
    """Return a list of all the tensors which are input or output of an op in
    the graph.
    """
    ts = []
    for op in self.nodes:
      ts += op.outputs
    return ts

  @property
  def version(self):
    """
    Returns a counter that goes up every time this graph is changed.
    """
    return self._version

  def _increment_version_counter(self):
    self._version += 1


class Tensor(object):
  """
  Surrogate object that represents an output of a Node. Corresponds roughly to 
  a tf.Tensor in the TensorFlow Python API, though serialized TensorFlow graphs 
  do not contain any separate objects that represent tensors.
  """
  def __init__(self, node, index, dtype, shape):
    """
    Args:
      node: pge.Node object that represents the graph node that produces this 
        tensor
      index: Output index of this tensor among the outputs of the specified node
      dtype: Data type of the tensor
      shape: Shape of the tensor
    """
    self._node = node
    self._index = index
    self._dtype = dtype
    self._shape = shape

  @property
  def operator(self):
      return self._node

  @property
  def value_index(self):
    return self._index

  @property
  def dtype(self):
    return self._dtype

  @property
  def shape(self):
    return self._shape

  @property
  def graph(self):
    """Returns the `pge.Graph` object representing the graph in which the
    operator that produces this tensor resides."""
    return self._node.graph

  @property
  def consumers(self):
    """Returns the `pge.Node` objects representing the ops that consume the
    tensor that this object represents."""
    # TODO: Maintain a lookup table of graph edges.
    # For now we do linear search for correctness.
    ret = []
    for n in self.graph.nodes:
      if self in n.inputs:
        ret.append(n)

    return ret


class Node(object):
  """
  Public API for interacting with graph nodes
  """
  def __init__(self, graph: Graph, name: str, outputs: List[Tensor]):
    """
    This constructor should only be called by subclasses.
    """
    self._graph = graph
    self._name = name
    self._outputs = outputs

  @property
  def name(self):
    """
    Returns:
       Unique name of the operator that this Node represents
    """
    return self._name

  @property
  def graph(self):
    """
    Returns:
      `pge.Graph` object representing the graph in which this Node resides.
    """
    return self._graph

  @property
  def outputs(self):
    """
    Returns:
      Tuple (i.e. immutable list) of `pge.Tensor` objects representing the
      current outputs of this node. Note that this tuple does not change if
      the underlying node is mutable and gets edited.
    """
    return tuple(self._outputs)

  def output(self, index: int):
    """
    Args:
      index: Index of an output of the node
    Returns:
      The Tensor corresponding to the indicated output of the node
    """
    return self._outputs[index]

  @property
  def inputs(self) -> Tuple[Tensor]:
    """
    Returns:
      Tuple (i.e. immutable list) of `pge.Tensor` objects representing the
      current inputs of this node.
    """
    raise NotImplementedError("This method should be implemented by "
                              "subclasses.")

  @property
  def control_inputs(self) -> Tuple['Node']:
    """
    Returns:
      Tuple (i.e. immutable list) of `pge.Node` objects representing the
      nodes that have control edges to this node.
    """
    raise NotImplementedError("This method should be implemented by "
                              "subclasses.")


class ImmutableNode(Node):
  """
  Wrapper for tf.NodeDef. Also maintains a pointer back to wrapper object for 
  the original graph.
  """

  def __init__(self, graph: Graph, node_def: tf.NodeDef,
               outputs_list: List[Tuple[tf.DType, tf.shape]]):
    """
    Args:
      graph: pge.Graph object that represents the parent graph
      node_def: tf.NodeDef protobuf 
      outputs_list: List of (type, shape) pairs that describe the outputs of 
        this node
    """
    Node.__init__(self, graph, node_def.name,
                  [Tensor(self, i, outputs_list[i][0], outputs_list[i][1])
                   for i in range(len(outputs_list))])
    self._node_def = node_def

  @Node.inputs.getter
  def inputs(self) -> Tuple[Tensor]:
    # Input names in the protobuf take three forms:
    #   "^node_name" --> Control input from indicated node
    #   "node_name" --> Input from output number 0 of indicated node
    #   "node_name:ix" --> Input from output number <ix> of indicated node
    # Start by filtering out the control inputs and turning "node_name" into
    # "node_name:0".
    input_names = [_canonicalize_output_name(n) for n in self._node_def.input
                   if not n.startswith("^")]
    input_tensors = []
    for name in input_names:
      # Name is in form "node:output number"
      node_name, output_ix_name = name.split(":")
      output_ix = int(output_ix_name)
      input_tensors.append(self.graph[node_name].output(output_ix))
    return tuple(input_tensors)

  @Node.control_inputs.getter
  def control_inputs(self) -> Tuple[Node]:
    # Control inputs start with "^". Skip everything else and strip off the
    # leading caret character
    control_input_names = [n[1:] for n in self._node_def.input
                           if n.startswith("^")]
    return tuple([self.graph[name] for name in control_input_names])


class MutableNode(Node):
  """
  Wrapper for a change to a graph that will add a node. Accumulates the
  parameters of the node to be added and can produce an appropriate
  tf.NodeDef protobuf on demand.
  """

  def __init__(self, graph: Graph, name: str, op: str):
    """
    This constructor should only be called form Graph.add_node().

    Args:
      graph: The graph that this node is to be added to. The caller is
        responsible for adding the node to the graph.
      name: Name of the new node to add
      op: Name of the operation that the new node will perform
    """
    Node.__init__(self, graph, name, [])
    self._op = op
    self._attributes = []

  def add_attr(self, key: str, value):
    """Add a single attribute to the underlying NodeDef's attr list.

    Args:
      key: Name of the attribute. Must be unique.
      value: Value to put in place for the attribute. Must be one of the
        following types:
        * tf.DType
        * tf.TensorShape
    """
    if key in self._attr_names():
      raise ValueError("Already have an attribute called '{}'".format(key))
    self._attributes.append((key, value))

  def _attr_names(self):
    return [a[0] for a in self._attributes]

  @Node.inputs.getter
  def inputs(self) -> Tuple[Tensor]:
    raise NotImplementedError("This method not yet implemented.")

  @Node.control_inputs.getter
  def control_inputs(self) -> Tuple[Node]:
    raise NotImplementedError("This method not yet implemented.")


# Stuff below this line is private to this file.


def _canonicalize_output_name(name: str):
  """
  Args:
    name: Name for an op output as it would appear in the protocol buffer
      representation of a an operator graph
  Returns:
    A name in the form "<op name>:<output index>"
  """
  if ":" in name:
    return name
  else:
    return name + ":0"


def _decode_graph(graph_def):
  """
  Use public TensorFlow APIs to decode the important information that is not 
  explicitly stored in the GraphDef proto, but which must be inferred from the 
  GraphDef in conjunction with additional data structures that TensorFlow 
  generally keeps to itself. 

  Args:
    graph_def: tf.GraphDef protobuf that represents a TensorFlow operator graph.
      This graph must be runnable on the current version of TensorFlow; 
      otherwise some of the type inference operations that this function 
      performs will fail.

  :returns: A tuple (output_map, ...)
  where:
    output_map is a map from operator name to a list of (type, shape) pairs 
      that describe in turn each of the outputs of said operator.
  """
  # The information in a NodeDef is not sufficient to determine output type
  # information. For that kind of type inference, you need access to the
  # corresponding OpDef protos. Unfortunately there is not a public API that
  # allows for OpDef lookup. So instead we instantiate the graph that
  # graph_def describes. This approach makes things easier, but there will be
  # a reduction in forwards compatibility, because import_graph_def() does a
  # lot of sanity checks that aren't necessary when rewriting a graph_def.
  temp_graph = tf.Graph()
  # tf.import_graph_def() insists on prepending a prefix onto the name of every
  # node it imports. There's no way to turn this functionality off, so use a
  # known prefix.
  _PREFIX = "pge_import"
  with temp_graph.as_default():
    tf.import_graph_def(graph_def, name=_PREFIX)
  output_map = {op.name[(len(_PREFIX) + 1):]: [(t.dtype, t.shape)
                                               for t in op.outputs]
                for op in temp_graph.get_operations()}
  return output_map
