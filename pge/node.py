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

from copy import deepcopy
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Iterable, Any

from pge import graph
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
  def __init__(self, g: 'graph.Graph', node_id: int, name: str,
               op_name: str, outputs: List[tensor.Tensor],
               device: str):
    """
    This constructor should only be called by subclasses.
    """
    self._graph = g
    self._id = node_id
    self._name = name
    self._op_name = op_name
    self._outputs = outputs
    self._device = device

  @property
  def name(self):
    """
    Returns:
       Unique name of the node that this Node represents
    """
    return self._name

  @property
  def op_type(self):
    """
    Returns:
      Name of the TensorFlow op type that this Node represents
    """
    return self._op_name

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

  @property
  def device(self):
    """
    Returns:
      TensorFlow device placement string desribing where this node should be
      placed, or None to specify use of the default device.
    """
    return self._device

  def to_node_def(self):
    """
    Returns:
        A copy of the contents of this node as a NodeDef proto. The returned
        proto will *not* change if this node is changed after the call, and
        vice versa.
    """
    raise NotImplementedError("This method should be implemented by "
                              "subclasses.")

  def get_attr(self, key: str) -> Any:
    """
    Retrieve the value of an attribute by name.

    Args:
      key: Key under which the node's attribute is stored

    Returns:
      Current value of the attribute as an appropriate native Python type
      (NOT a `tf.AttrValue` protobuf) or None if no value was found.

    Raises:
      ValueError if the indicated key does not have an attribute associated
      with it.
    """
    raise NotImplementedError("This method should be implemented by "
                              "subclasses.")

  def get_attr_keys(self) -> Tuple[str]:
    """
    Returns:
      Tuple (immutable list) of the keys of all attributes currently present
      in the node
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
    Node.__init__(self, g, node_id=node_id, name=node_def.name,
                  op_name=node_def.op,
                  outputs=[tensor.Tensor(self, i, outputs_list[i][0],
                                         outputs_list[i][1])
                           for i in range(len(outputs_list))],
                  device=node_def.device)
    self._node_def = node_def

  @Node.inputs.getter
  def inputs(self) -> Tuple[tensor.Tensor]:
    # Regenerate each time for now.
    return tuple(_decode_inputs(self._node_def.input, self._graph))

  @Node.control_inputs.getter
  def control_inputs(self) -> Tuple[Node]:
    # For now, regenerate every time
    return tuple(_decode_control_inputs(self._node_def.input, self._graph))

  def get_attr(self, key: str):
    if key not in self._node_def.attr:
      raise ValueError("Node {} does not have an attribute "
                       "under key '{}'".format(self, key))
    return _attr_value_to_python_type(self._node_def.attr[key])

  def get_attr_keys(self) -> Tuple[str]:
    return tuple(self._node_def.attr)

  def to_node_def(self):
    return deepcopy(self._node_def)


class MutableNode(Node):
  """
  Wrapper for a change to a graph that will add a node. Accumulates the
  parameters of the node to be added and can produce an appropriate
  tf.NodeDef protobuf on demand.
  """

  def __init__(self, g: 'graph.Graph', node_id: int, name: str, op_name: str,
               device: str = ""):
    """
    This constructor should only be called from methods of the Graph
    class.

    Args:
      g: The graph that this node is to be added to. The caller is
        responsible for adding the node to the graph.
      node_id: Unique (within the parent graph) integer identifier for the node
      name: Name of the new node to add
      op_name: Name of the operation that the new node will perform
      device: TensorFlow device specification string indicating where this node
        should be located. Default value of "" means "use the default device"
    """
    Node.__init__(self, g, node_id=node_id, name=name,
                  op_name=op_name, outputs=[], device=device)
    self._attributes = []
    self._inputs = []
    self._control_inputs = []

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

  def get_attr(self, key: str):
    # self._attributes is a list of (key, value) pairs
    matches = [p[1] for p in self._attributes if p[0] == key]
    if 0 == len(matches):
      raise ValueError("Node {} does not have an attribute "
                       "under key '{}'".format(self, key))
    elif len(matches) > 1:
      raise ValueError("Node {} has more than one attribute "
                       "under key '{}'".format(self, key))
    ret = matches[0]
    if isinstance(ret, tf.AttrValue):
      return _attr_value_to_python_type(ret)
    else:
      return ret

  def get_attr_keys(self) -> Tuple[str]:
    return tuple([p[0] for p in self._attributes])

  def clear_attrs(self):
    """
    Remove any attributes that are attached to this node.
    """
    self._attributes.clear()

  def _attr_names(self):
    return [a[0] for a in self._attributes]

  @Node.inputs.getter
  def inputs(self) -> Tuple[tensor.Tensor]:
    return tuple(self._inputs)

  def replace_input(self, index: int, new_input: tensor.Tensor):
    """
    Replace an existing input of this node with the specified tensor. Roughly
    equivalent to `tf.Operator._update_input()`.

    Does NOT change the output type or shape of the node. You may need to
    call `infer_outputs()` or `set_outputs_from_pairs()` to update the node's
    output type information after changing an input with this method.

    Args:
      index: Index of input to replace. Note that this index is an offset into
        the data inputs, NOT the control inputs, of the node.
      new_input: The replacement input

    Raises:
      IndexError if index does not correspond to an existing input.
    """
    if index < 0 or index >= len(self._inputs):
      raise IndexError("Received input index {}, but node has {} "
                       "inputs".format(index, len(self._inputs)))
    self._inputs[index] = new_input
    self._graph.increment_version_counter()

  def set_inputs(self, new_inputs: Iterable[tensor.Tensor]):
    """
    Set all inputs at once, removing anything that was there previously.

    Args:
      new_inputs: Iterable of `Tensor` objects in this node's parent graph
    """
    for t in new_inputs:
      if t.graph != self.graph:
        raise ValueError("Tensor {} points to graph {}, but this node is in a "
                         "different graph {}".format(t, t.graph, self.graph))
    self._inputs = list(new_inputs)
    self._graph.increment_version_counter()  # New edges added to graph

  def set_control_inputs(self, new_control_inputs: Iterable[Node]):
    """
    Set all control inputs at once, removing anything that was there
    previously.

    Args:
      new_control_inputs: Iterable of `Node` objects in this node's parent graph
    """
    self._control_inputs = list(new_control_inputs)

  def set_outputs_from_pairs(self, new_outputs: Iterable[Tuple[tf.DType,
                                                               tf.shape]]):
    """
    Set all outputs at once, removing anything that was there previously.

    Note that information about outputs is not stored in the serialized graph.
    When instantiating a serialized graph, TensorFlow will use its own shape
    inference to infer the number, type, and shape of the node's outputs.

    Args:
      new_outputs: Iterable of (dtype, shape) pairs that describe the outputs
    """
    self._outputs = []
    i = 0
    for (dtype, shape) in new_outputs:
      self._outputs.append(tensor.Tensor(self, i, dtype, shape))
      i += 1
    self._graph.increment_version_counter()  # Just in case

  def infer_outputs(self):
    """
    Use TensorFlow's shape and dtype inference to determine the number of
    outputs as well as their shapes and dtypes, based on the node's op type
    string, its attribute values, and what inputs are connected to it.

    Inference will only function properly if the currently-loaded version of
    TensorFlow knows about the specified op type and the current
    configuration of this op's inputs is compatible with the combination of
    op type string and parameters.

    Overwrites the previous value of the `outputs` property.

    Raises:
      TBD
    """
    # TF lack a supported API for invoking shape inference directly,
    # so we instantiate a dummy graph and create a dummy Operation object
    temp_graph = tf.Graph()
    with temp_graph.as_default():
      input_placeholders = [tf.placeholder(shape=t.shape, dtype=t.dtype) for
                            t in self._inputs]
      # See the docs for tf.Operation for important notes about the semantics
      # of each arg to the following constructor.
      dummy_op = tf.Operation(self.to_node_def(), temp_graph,
                              inputs=input_placeholders)
      self.set_outputs_from_pairs([(o.dtype, o.shape)
                                   for o in dummy_op.outputs])
      # set_outputs_from_pairs() increments the version counter, so we don't
      # need to. Also, we haven't added edges to the graph until these
      # outputs are connected to another node's inputs.

  def set_inputs_from_strings(self, new_inputs: Iterable[str],
                              set_control_inputs: bool = True):
    """
    Set all input at once, converting TensorFlow string-format inputs into
    `Tensor` objects. All nodes referenced in the input strings must be
    present in the parent graph.

    Args:
      new_inputs: Input description strings in the format that they appear in a
       `tf.NodeDef` protocol buffer.
      set_control_inputs: If True, replace existing control inputs for this
        node with any control inputs specified in the input strings.
        Otherwise , this method will ignore any strings that describe control
        inputs.
    """
    self._inputs = _decode_inputs(new_inputs, self._graph)
    if set_control_inputs:
      self._control_inputs = _decode_control_inputs(new_inputs, self._graph)
    self._graph.increment_version_counter()  # New edges added to graph

  @Node.control_inputs.getter
  def control_inputs(self) -> Tuple[Node]:
    return tuple(self._control_inputs)

  def to_node_def(self):
    ret = tf.NodeDef()
    ret.name = self.name
    ret.op = self.op_type
    for input_tensor in self.inputs:
      ret.input.append(input_tensor.name)
    for control_input_node in self.control_inputs:
      ret.input.append("^" + control_input_node.name)
    ret.device = self.device
    for (attr_name, attr_value) in self._attributes:
      # Funky syntax for setting a field of a union in a protobuf
      ret.attr[attr_name].CopyFrom(_python_type_to_attr_value(attr_value))
    return ret

  def set_device(self, device: str):
    self._device = device


################################################################################
# Stuff below this line is private to this file.


def _canonicalize_output_name(name: str):
  """
  Args:
    name: Name for an op output as it would appear in the protocol buffer
      representation of a an node graph
  Returns:
    A name in the form "<op name>:<output index>"
  """
  if ":" in name:
    return name
  else:
    return name + ":0"


def _decode_inputs(inputs: Iterable[str], g: 'graph.Graph') -> List[
  tensor.Tensor]:
  """
  Extract and decode the inputs in a list of TensorFlow input specification
  strings.

  Skips over control inputs.

  Args:
    inputs: List of strings specifying data and/or control inputs,
      as serialized in `tf.NodeDef` protocol buffers.
    g: Reference to a `Graph` object that must have nodes corresponding
      to all inputs in the inputs list.

  Returns:
    A list of `Tensor` objects corresponding to each of the specified inputs.
  """
  # Input names in the protobuf take three forms:
  #   "^node_name" --> Control input from indicated node
  #   "node_name" --> Input from output number 0 of indicated node
  #   "node_name:ix" --> Input from output number <ix> of indicated node
  # Start by filtering out the control inputs and turning "node_name" into
  # "node_name:0".
  input_names = [_canonicalize_output_name(n) for n in inputs
                 if not n.startswith("^")]
  input_tensors = []
  for name in input_names:
    # Name is in form "node:output number"
    node_name, output_ix_name = name.split(":")
    output_ix = int(output_ix_name)
    input_tensors.append(g[node_name].output(output_ix))
  return input_tensors


def _decode_control_inputs(inputs: Iterable[str], g: 'graph.Graph') -> List[
  Node]:
  """
  Extract and decode the control inputs in a list of TensorFlow input
  specification strings.

  Skips data inputs.

  Args:
     inputs: List of strings specifying data and/or control inputs,
      as serialized in `tf.NodeDef` protocol buffers.
    g: Reference to a `Graph` object that must have nodes corresponding
      to all inputs in the inputs list.

  Returns:
    A list of `Node` objects corresponding to each of the control inputs.
  """
  # Control inputs start with "^". Skip everything else and strip off the
  # leading caret character
  control_input_names = [n[1:] for n in inputs if n.startswith("^")]
  return [g[name] for name in control_input_names]


def _python_type_to_attr_value(value: Any) -> tf.AttrValue:
  """
  Convert a Python object or scalar value to a TensorFlow `tf.AttrValue`
  protocol buffer message.

  Args:
    value: Python object to be converted

  Returns:
    An AttrValue object that wraps the contents of `value` in the most
    appropriate way available.
  """
  # TODO(frreiss): Handle AttrValues that are lists
  if isinstance(value, tf.AttrValue):
    # TODO(frreiss): Should this case result in an error?
    return value
  # Scalar types, in the order they appear in the .proto file
  elif isinstance(value, str):
    return tf.AttrValue(s=tf.compat.as_bytes(value))
  elif isinstance(value, int):
    return tf.AttrValue(i=value)
  elif isinstance(value, float):
    return tf.AttrValue(f=value)
  elif isinstance(value, bool):
    return tf.AttrValue(b=value)
  elif isinstance(value, tf.DType):
    return tf.AttrValue(type=value.as_datatype_enum)
  elif isinstance(value, tf.TensorShape):
    return tf.AttrValue(shape=value.as_proto())
  elif isinstance(value, np.ndarray):
    return tf.AttrValue(tensor=tf.make_tensor_proto(values=value))
  # TODO(frreiss): Populate the "func" and "placeholder" fields of the union
  #  here
  else:
    raise ValueError("Don't know how to convert a {} to "
                     "tf.AttrValue".format(type(value)))


def _attr_value_to_python_type(attr_value: tf.AttrValue) -> Any:
  """
  Inverse of _python_type_to_attr_value().

  Args:
    attr_value: Protocol buffer version of a node's attribute value

  Returns:
    A Python object or built-in type corresponding to the field in
    `attr_value` that is in use.
  """
  # TODO(frreiss): Handle AttrValues that are lists
  if attr_value.HasField("s"):          # str
    # TODO(frreiss): Should we return the binary value here?
    return tf.compat.as_str(attr_value.s)
  elif attr_value.HasField("i"):        # int
    return attr_value.i
  elif attr_value.HasField("f"):        # float
    return attr_value.f
  elif attr_value.HasField("b"):        # bool
    return attr_value.b
  elif attr_value.HasField("type"):     # DType
    return tf.DType(attr_value.type)
  elif attr_value.HasField("shape"):    # TensorShape
    # Undocumented behavior of public API: tf.TensorShape constructor accepts
    # a TensorShapeProto.
    return tf.TensorShape(attr_value.shape)
  elif attr_value.HasField("tensor"):   # TensorProto
    return tf.make_ndarray(attr_value.tensor)
  # TODO(frreiss): Convert the "func" and "placeholder" fields of the union
  #  here
  else:
    raise ValueError("Don't know how to convert AttrValue {} to "
                     "a Python object".format(attr_value))

