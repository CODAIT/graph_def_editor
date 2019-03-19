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
import sys
if sys.version >= '3':
  from typing import Tuple, List, Iterable, Any, AbstractSet

from graph_def_editor import graph, tensor, util

# Magical attribute name that TensorFlow uses to store colocation groups.
# See colocation_groups property below for more information.
_COLOCATION_ATTR_NAME = "_class"

# Magical prefix that TensorFlow appends to node names to create colocation
# group names.
_COLOCATION_PREFIX = "loc:@"

# Magical attribute name that TensorFLow uses to store shape information.
# Note that TensorFlow will treat the value of this field as truth and will
# skip shape inference if it is present.
_OUTPUT_SHAPES_ATTR_NAME = "_output_shapes"

__all__ = [
    "Node",
]


class Node(object):
  """
  Mutable surrogate for a `tf.NodeDef` protocol buffer message.
  Accumulates the parameters of the node and can produce an appropriate
  tf.NodeDef protobuf on demand.
  """
  def __init__(self,
               g, # type: graph.Graph
               node_id, # type: int
               name, # type: int
               op_name, # type: str
               device = "" # type: str
               ):
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
    self._graph = g
    self._id = node_id
    self._name = name
    self._op_name = op_name
    self._device = device
    self._attributes = []  # List[Tuple[str,Any]]
    self._inputs = []
    self._outputs = None  # List[Tensor]
    self._control_inputs = []
    self._colocation_groups = []  # List[str]
    self._collection_names = set()  # Set[str]

  def __repr__(self):
    # type: () -> str
    return "Node[{}|{}]".format(self.name, self.op_type)

  @property
  def name(self):
    # type: () -> str
    """
    Returns:
       Unique name of the node that this Node represents
    """
    return self._name

  def _change_name(self, new_name):
    # type: (str) -> None
    """
    THIS METHOD SHOULD ONLY BE CALLED BY THE PARENT GRAPH

    Changes this node's `name` attribute WITHOUT UPDATING THE PARENT GRAPH.

    Args:
      new_name: New value for the `name` attribute.
    """
    self._name = new_name

  @property
  def op_type(self):
    # type: () -> str
    """
    Returns:
      Name of the TensorFlow op type that this Node represents
    """
    return self._op_name

  def change_op_type(self, new_op_type):
    # type: (str) -> None
    """
    Change the op type of this node. Does NOT rerun shape or type inference.

    Args:
      new_op_type: New string value for the operator type. Should correspond
        to the name of a TensorFlow op, although this method does not validate
        the string.
    """
    self._op_name = new_op_type

  @property
  def graph(self):
    # type: () -> graph.Graph
    """
    Returns:
      `gde.Graph` object representing the graph in which this Node resides.
    """
    return self._graph

  @property
  def id_in_graph(self):
    # type: () -> int
    """
    Returns this node's unique integer id within the parent graph. Useful for
    sorting nodes in an arbitrary but consistent order.
    """
    return self._id

  def _remove_from_graph(self):
    # type: () -> None
    """
    THIS METHOD TO BE CALLED ONLY BY THE PARENT GRAPH.

    Sets this node's graph pointer to None. DOES NOT UPDATE POINTERS TO THIS
    NODE FROM THE PARENT GRAPH.
    """
    self._graph = None
    # Don't need to update output tensors because they don't store a pointer
    # to the graph, only to the node

  @property
  def outputs(self):
    # type: () -> Tuple[tensor.Tensor]
    """
    Returns:
      Tuple (i.e. immutable list) of `gde.Tensor` objects representing the
      current outputs of this node. Note that this tuple does not change if
      the underlying node is mutable and gets edited.
    """
    if self._outputs is None:
      raise ValueError("Outputs of {} have not been set".format(self))
    return tuple(self._outputs)

  def output(self, index):
    # type: (int) -> tensor.Tensor
    """
    Args:
      index: Index of an output of the node
    Returns:
      The Tensor corresponding to the indicated output of the node
    """
    if self._outputs is None:
      raise ValueError("Outputs have not been set")
    return self._outputs[index]

  @property
  def inputs(self):
    # type: () -> Tuple[tensor.Tensor]
    """
    Returns:
      Tuple (i.e. immutable list) of `gde.Tensor` objects representing the
      current inputs of this node. Note that the returned value is immutable
      for a reason. Do not attempt to modify it.
    """
    return tuple(self._inputs)

  def replace_input(self, index, new_input):
    # type: (int, tensor.Tensor) -> None
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

  def set_inputs(self, new_inputs):
    # type: (Iterable[tensor.Tensor]) -> None
    """
    Set all inputs at once, removing anything that was there previously.

    Args:
      new_inputs: Iterable of `gde.Tensor` objects in this node's parent graph
    """
    for t in new_inputs:
      if t.graph != self.graph:
        raise ValueError("Tensor {} points to graph {}, but this node is in a "
                         "different graph {}".format(t, t.graph, self.graph))
    self._inputs = list(new_inputs)
    self._graph.increment_version_counter()  # New edges added to graph

  @property
  def control_inputs(self):
    # type: () -> Tuple[Node]
    """
    Returns:
      Tuple (i.e. immutable list) of `gde.Node` objects representing the
      nodes that have control edges to this node.
    """
    return tuple(self._control_inputs)

  @property
  def device(self):
    # type: () -> str
    """
    Returns:
      TensorFlow device placement string describing where this node should be
      placed, or None to specify use of the default device.
    """
    return self._device

  @device.setter
  def device(self, value):
    # type: (str) -> None
    self._device = value

  @property
  def colocation_groups(self):
    # type: () -> List[str]
    """
    **A word about colocation groups:**

    TensorFlow constrains some operators to be located on the same device
    as each other. The mechanism that TensorFlow uses to enforce these
    constraints is poorly documented, but I've pieced together the following
    by reading through source code.

    Every TensorFlow operator belongs to one or more colocation groups. Each
    such group has a name that takes the form "loc:@<operator name>",
    where <operator name> is the full name of an operator in the same graph.
    By default, every operator is a member of its own colocation group. An op
    can also be a member of any number of other ops' groups.

    At graph creation time, some internal TensorFlow functions use the
    `tf.colocate_with()` context manager (defined in
    `tensorflow/python/framework/ops.py`) to add colocation constraints to
    generated ops.  At runtime, TensorFlow's placement algorithms perform the
    necessary set-cover computations to force every member of every
    colocation group to be on the same device.

    When TensorFlow serializes a graph, the system shoehorns information about
    colocation into the `tf.NodeDef` protocol buffer messages that make up
    the `tf.GraphDef`. The implicit membership of an operator in its own
    colocation group is *not* stored anywhere in the NodeDef; the user just
    needs to know that every node "N" is a member of the colocation group
    "loc:@N". Other additional colocation groups are stored in an attribute
    called "_class" (a name apparently chosen to confuse the enemy) as a list
    of strings. Here's an example from a call to `tf.while_loop()`:

    ```
    node {
      name: "while/Switch"
      op: "Switch"
      input: "while/Merge"
      input: "while/LoopCond"
      attr {
        key: "T"
        value {
          type: DT_INT32
        }
      }
      attr {
        key: "_class"
        value {
          list {
            s: "loc:@while/Merge"
          }
        }
      }
    }
    ```

    This class, `Node`, internally stores colocation groups as a list of node
    names. Currently, we do *not* store the node's own implicit colocation
    group in this list. The `to_node_def()` method generates the "_class"
    attr as appropriate.

    Some important notes:
    * TensorFlow will refuse to load a `GraphDef` if any nodes contain
      references to nonexistent nodes in their collocation group attributes.
      When renaming a node, be sure to rename any colocation groups that
      reference the node.
    * In principle, it should be possible to store things in the "_class"
      attribute other than colocation groups; simply avoid using strings that
      start with the magic prefix "loc:@". In practice, using the "_class"
      attribute in this way is a bad idea, and this class will raise an
      exception if you attempt to do so.
    * It's unclear what happens if there are conflicting "device" directives in
      nodes that share a colocation group. It's best to ensure that that
      situation doesn't happen.
    * Yes, it's supposed to be spelled "collocate"/"collocation", but the
      misspelling "colocate" is thoroughly entrenched in the tech community.

    Returns:
      Names of other Nodes that this Node is constrained to be collocated
      with. *Does NOT return this Node's "default" colocation group.*
      *Does NOT return node names prefixed with "loc:@".
      Modifications to the returned tuple will NOT be reflected in the Node.
      Use `add_colocation_group` and the setter for this property if you wish
      to modify a node's colocation group information.
    """
    return tuple(self._colocation_groups)

  @colocation_groups.setter
  def colocation_groups(self):
    # type: (Iterable[str]) -> None
    """
    Setter for the `colocation_groups` property.

    @param value: New set of colocation groups, superseding the current set.

    Raises:
      ValueError if any of the colocation groups reference
    """
    for s in value:
      if not self._graph.contains_node(s):
        raise ValueError("Graph does not contain a node with name '{}'".format(
          s))
    self._colocation_groups = value
    # Invalidate any cached information that the parent Graph may have
    # generated about colocation constraints.
    self.graph.increment_version_counter()

  def add_colocation_group(self, head_node_name, validate = True):
    # type: (str, bool) -> None
    """
    Add a new colocation group to this Node.

    See the docstring for the `colocation_groups` property for more
    information about colocation groups.

    Args:
      head_node_name: Name of the node in the graph that serves as the
        "primary" node in the group. By convention, the name of the group will
        be "loc:@<head_node_name>".
      validate: If True, verify that the target node exists in the parent graph.

    Raises:
      ValueError if there is a problem with `head_node_name`
    """
    # TODO(frreiss): Invalidate any cached information that the parent Graph
    #  may have generated about colocation constraints.
    if validate and not self._graph.contains_node(head_node_name):
        raise ValueError("Graph does not contain a node with name '{}'".format(
          head_node_name))
    if head_node_name in self._colocation_groups:
      raise ValueError("Already have colocation group with '{}'".format(
        head_node_name))
    self._colocation_groups.append(head_node_name)

  def to_node_def(self, target = None, add_shapes = True):
    # type: (tf.NodeDef, bool) -> tf.NodeDef
    """
    Args:
      target: optional preallocated, empty NodeDef object to fill in. If not
        provided, this method will allocate a new `tf.NodeDef` object.
      add_shapes: If True, add the special "_output_shapes" attribute with
        output shape information from this Node's output metadata.
    Returns:
        A copy of the contents of this node as a NodeDef proto. The returned
        proto will *not* change if this node is changed after the call, and
        vice versa.
    """
    if target is None:
      target = tf.NodeDef()
    target.name = self.name
    target.op = self.op_type
    for input_tensor in self.inputs:
      target.input.append(input_tensor.name)
    for control_input_node in self.control_inputs:
      target.input.append("^" + control_input_node.name)
    target.device = self.device
    for (attr_name, attr_value) in self._attributes:
      # Funky syntax for setting a field of a union in a protobuf
      target.attr[attr_name].CopyFrom(
        util.python_type_to_attr_value(attr_value))
    if len(self._colocation_groups) > 0:
      # Serialize colocation groups. See docstring in getter for
      # colocation_groups property for more information.
      transformed_names = [_COLOCATION_PREFIX + name
                           for name in self._colocation_groups]
      target.attr[_COLOCATION_ATTR_NAME].CopyFrom(
        util.python_type_to_attr_value(transformed_names)
      )
    if add_shapes and self._outputs is not None and len(self._outputs) > 0:
      shapes_list = [t.shape for t in self._outputs]
      target.attr[_OUTPUT_SHAPES_ATTR_NAME].CopyFrom(
        util.python_type_to_attr_value(shapes_list)
      )
    return target

  def get_attr(self, key):
    # type: (str) -> Any
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
      return util.attr_value_to_python_type(ret)
    else:
      return ret

  def has_attr(self, key):
    # type: (str) -> bool
    """
    Args:
      key: String name of a potential attribute

    Returns True if the node has an attribute under the indicated key
    """
    return key in self._attributes

  def get_attr_keys(self):
    # type: () -> Tuple[str]
    """
    Returns:
      Tuple (immutable list) of the keys of all attributes currently present
      in the node
    """
    return tuple([p[0] for p in self._attributes])

  def add_attr(self,
               key, # type: str
               value, # type: Any
               validate_colocation_groups = False # type: bool
               ):
    # type: (...) -> None
    """Add a single attribute to the underlying NodeDef's attr list.

    If you use this method to set the special "_class" attribute,
    will redirect to a call to the setter for the `colocation_groups`
    property.

    If you use this method to set the special "_output_shapes" attribute,
    the method will redirect to a call to the set_outputs_from_pairs()
    method.

    Args:
      key: Name of the attribute. Must be unique.
      value: Value to put in place for the attribute. Can be a Python type or a
        TensorFlow protocol buffer wrapper class.
      validate_colocation_groups: If True and this method is setting colocation
        groups via the special '_class' attribute, raise an exception if the
        primary node of the colocation group does not exist.
    """
    if key == _COLOCATION_ATTR_NAME:
      # Special magic key name for colocation groups; see docstring for
      # colocation_groups property.
      if len(self.colocation_groups) > 0:
        raise ValueError("Tried to set special '{}' attribute when the "
                         "Node already has colocation "
                         "groups".format(_COLOCATION_ATTR_NAME))
      for group_name in _validate_colocation_group_attr(value):
        self.add_colocation_group(group_name,
                                  validate=validate_colocation_groups)
    elif key == _OUTPUT_SHAPES_ATTR_NAME:
      # Special magic key name for output shapes.
      new_shapes = _validate_output_shapes_attr(value)
      self._update_shapes(new_shapes)
    elif key in self._attr_names():
      raise ValueError("Already have an attribute called '{}'".format(key))
    else:
      # Make sure attributes appear in protobuf in the order added
      self._attributes.append((key, value))

  def _update_shapes(self, new_shapes):
    # type: (List[tf.TensorShape]) -> None
    """
    Put a set of output shapes in place without changing dtypes. Raises an
    error if doing so would change the number of outputs. Sets dtypes to None
    if no output information is present.
    """
    if self._outputs is None:
      pairs = [(None, s) for s in new_shapes]
      self.set_outputs_from_pairs(pairs)
    else:
      if len(new_shapes) != len(self._outputs):
        raise ValueError("Attempted to put in place {} output shapes, "
                         "but node has {} outputs.".format(len(new_shapes),
                                                           len(self._outputs)))
      # Update shapes in place to avoid creating new Tensor objects.
      # If we created new Tensor objects, we would need to update all the
      # downstream ops that used those Tensors as inputs.
      for i in range(len(new_shapes)):
        self._outputs[i].shape = new_shapes[i]

  def replace_attr(self,
                   key, # type: str
                   value, # type: Any
                   validate_colocation_groups = False # type: bool
                   ):
    # type: (...) -> None
    """
    Replace an existing attribute in the underlying NodeDef's attr list,
    without changing the order of the list.

    If you use this method to set the special "_class" attribute,
    will redirect to a call to the setter for the `colocation_groups`
    property.

    Args:
      key: Name of the attribute. Must be unique.
      value: Value to put in place for the attribute. Can be a Python type or a
        TensorFlow protocol buffer wrapper class.
      validate_colocation_groups: If True and this method is setting colocation
        groups via the special '_class' attribute, raise an exception if the
        primary node of the colocation group does not exist.
    """
    if key == _COLOCATION_ATTR_NAME:
      # Special magic key name for colocation groups; see docstring for
      # colocation_groups property.
      if len(self.colocation_groups) == 0:
        raise ValueError("Tried to replace special '{}' attribute when the "
                         "Node does not have any colocation "
                         "groups".format(_COLOCATION_ATTR_NAME))
      for group_name in self._validate_colocation_group_attr(value):
        self.add_colocation_group(group_name,
                                  validate=validate_colocation_groups)
    elif key == _OUTPUT_SHAPES_ATTR_NAME:
      # Special magic key name for output shapes.
      new_shapes = _validate_output_shapes_attr(value)
      self._update_shapes(new_shapes)
    elif key not in self._attr_names():
      raise ValueError("{} has no attribute called '{}'".format(self, key))
    else:
      for i in range(len(self._attributes)):
        if self._attributes[i][0] == key:
          self._attributes[i] = (key, value)
          break

  def clear_attrs(self):
    # type: () -> None
    """
    Remove any attributes that are attached to this node.
    """
    self._attributes = []

  def _attr_names(self):
    # type: () -> List[str]
    return [a[0] for a in self._attributes]

  def set_control_inputs(self, new_control_inputs):
    # type: (Iterable[Node]) -> None
    """
    Set all control inputs at once, removing anything that was there
    previously.

    Args:
      new_control_inputs: Iterable of `Node` objects in this node's parent graph
    """
    self._control_inputs = list(new_control_inputs)

  def set_outputs_from_pairs(self, new_outputs):
    # type: (List[Tuple[tf.DType, tf.TensorShape]) -> None
    """
    Set all outputs at once, removing anything that was there previously.

    Note that information about outputs is not stored in the serialized graph.
    When instantiating a serialized graph, TensorFlow will use its own shape
    inference to infer the number, type, and shape of the node's outputs.

    Args:
      new_outputs: List of (dtype, shape) pairs that describe the outputs
    """
    if self._outputs is not None and len(new_outputs) != len(self._outputs):
      # TODO(frreiss): Implement changing the number of outputs. This
      #  implementation will require walking the graph and dealing with pointers
      #  to Tensors that don't exist.
      raise NotImplementedError("Attempted to change number of output tensors "
                                "on node {} from {} to {}. Changing the "
                                "number of output tensors is not currently "
                                "supported.".format(self.name,
                                                    len(self._outputs),
                                                    len(new_outputs)))
    elif self._outputs is None:
      self._outputs = [tensor.Tensor(self, i, None, None)
                       for i in range(len(new_outputs))]

    # At this point, self._outputs is initialized. Update dtypes and shapes
    # in place.
    for i in range(len(new_outputs)):
      self._outputs[i].dtype, self._outputs[i].shape = new_outputs[i]
    self._graph.increment_version_counter()  # Just in case

  def infer_outputs(self):
    # type: () -> None
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
    if self.op_type == "Assign":
      # SPECIAL CASE: Assign op takes a reference as input. Don't build up a
      # graph and invoke shape inference, because the APIs for references are
      # in flux. Instead, just trust the attributes.
      # First input is the reference, second is the value to put in place.
      # Assign op returns the reference that it just assigned to.
      input_ref = self._inputs[0]
      self.set_outputs_from_pairs([(input_ref.dtype, input_ref.shape)])
    else:
      # Common case: Use shape inference.
      # TF lacks a supported API for invoking shape inference directly,
      # so we instantiate a dummy graph and create a dummy Operation object.
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

        # TODO(frreiss): If this op has a "T" attribute, set that too.

  def set_inputs_from_strings(self, new_inputs, set_control_inputs = True):
    # type: (Iterable[str], bool) -> None
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

  @property
  def collection_names(self):
    # type: () -> AbstractSet[str]
    """
    Returns the names of all collections this node is a member of in the
    parent graph.
    """
    return frozenset(self._collection_names)

  def add_to_collection(self, collection_name):
    # type: (str) -> None
    """
    Add the node to the indicated collection.
    """
    if collection_name not in self._collection_names:
      self._collection_names.add(collection_name)
      # Invalidate any information the parent graph may have cached about
      # collections.
      self._graph.increment_version_counter()

  def remove_from_collections(self):
    # type: () -> None
    """
    Remove this node from amy collections that it is currently a member of.
    """
    self._collection_names = set()


################################################################################
# Stuff below this line is private to this file.


def _canonicalize_output_name(name):
  # type: (str) -> str
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


def _decode_inputs(inputs, # type: Iterable[str]
                   g # type: graph.Graph
  ):
  # type: (...) -> List[tensor.Tensor]
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


def _decode_control_inputs(inputs, # type: Iterable[str]
                           g # type: graph.Graph
                           ):
  # type: (...) -> List[Node]
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


def _validate_colocation_group_attr(value):
  # type: (Any) -> List[str]
  """Validate a potential value for the special "_class" attribute that
  holds collocation groups.

  Returns a list of node names that comprise the group."""
  if isinstance(value, tf.AttrValue):
    # Internal TF type; convert to iterable of Python strings
    if value.list.s is None:
      raise ValueError("Tried to set special '{}' attribute using "
                       "tf.AttrValue object, and the object's 'list.s' "
                       "attribute was not populated. Value: "
                       "'{}'".format(_COLOCATION_ATTR_NAME, str(value)))
    value = [tf.compat.as_str(s_i) for s_i in value.list.s]
  elif not isinstance(value, list) and not isinstance(value, tuple):
    raise ValueError("Tried to set special '{}' attribute with a type "
                     "other than list or tuple. Type is '{}' and value "
                     "is '{}'".format(_COLOCATION_ATTR_NAME, type(value),
                                      str(value)))
  ret = []
  for elem in value:
    if not elem.startswith(_COLOCATION_PREFIX):
      raise ValueError("Tried to set special '{}' attribute with "
                       "something other than a string starting with "
                       "'{}' (value used: "
                       "'{}')".format(_COLOCATION_ATTR_NAME,
                                      _COLOCATION_PREFIX, elem))
    ret.append(elem[len(_COLOCATION_PREFIX):])
  return ret


def _validate_output_shapes_attr(value):
  # type: (Any) -> List[tf.TensorShape]
  """
  Validate a potential value for the special "_output_shapes" attribute.

  Returns a list of output shapes extracted from the attribute value.
  """
  if isinstance(value, tf.AttrValue):
    if value.list.shape is None:
      raise ValueError("Tried to set special '{}' attribute using "
                       "tf.AttrValue object, and the object's 'list.shape' "
                       "attribute was not populated. Value: "
                       "'{}'".format(_OUTPUT_SHAPES_ATTR_NAME, str(value)))
    return [tf.TensorShape(shape_i) for shape_i in value.list.shape]
  else:
    raise ValueError("Tried to set special '{}' attribute with a type "
                     "other than a tf.AttrValue. Type is '{}' and value "
                     "is '{}'".format(_OUTPUT_SHAPES_ATTR_NAME, type(value),
                                      str(value)))
