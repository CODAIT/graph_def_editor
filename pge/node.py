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
"""Objects for representing nodes in a GraphDef proto."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Tuple, List

from pge import graph
from pge import node
from pge import tensor

__all__ = [
    "Node",
    "ImmutableNode",
    "MutableNode",
]


class Node(object):
  """
  Public API for interacting with graph nodes
  """
  def __init__(self, g: 'graph.Graph', node_id: int, name: str, outputs: List[
    tensor.Tensor]):
    """
    This constructor should only be called by subclasses.
    """
    self._graph = g
    self._id = node_id
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
  def inputs(self) -> Tuple[tensor.Tensor]:
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

  def __init__(self, g: 'graph.Graph', node_id: int, node_def: tf.NodeDef,
               outputs_list: List[Tuple[tf.DType, tf.shape]]):
    """
    Args:
      g: pge.Graph object that represents the parent graph
      node_id: Unique (within parent graph) integer identifier for this node
      node_def: tf.NodeDef protobuf 
      outputs_list: List of (type, shape) pairs that describe the outputs of 
        this node
    """
    Node.__init__(self, g, node_id, node_def.name,
                  [tensor.Tensor(self, i, outputs_list[i][0],
                                 outputs_list[i][1])
                   for i in range(len(outputs_list))])
    self._node_def = node_def

  @Node.inputs.getter
  def inputs(self) -> Tuple[tensor.Tensor]:
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

  def __init__(self, g: 'graph.Graph', node_id: int, name: str, op: str):
    """
    This constructor should only be called form Graph.add_node().

    Args:
      g: The graph that this node is to be added to. The caller is
        responsible for adding the node to the graph.
      node_id: Unique (within the parent graph) integer identifier for the node
      name: Name of the new node to add
      op: Name of the operation that the new node will perform
    """
    Node.__init__(self, g, node_id, name, [])
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
  def inputs(self) -> Tuple[tensor.Tensor]:
    raise NotImplementedError("This method not yet implemented.")

  @Node.control_inputs.getter
  def control_inputs(self) -> Tuple[Node]:
    raise NotImplementedError("This method not yet implemented.")


################################################################################
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


