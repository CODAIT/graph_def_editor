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

# TODO: Move this protobuf into this project so we don't depend on
#  tf.core.framework
from tensorflow.core.framework import variable_pb2

import sys
if sys.version >= '3':
  from graph_def_editor import graph
  from typing import AbstractSet, Union


__all__ = [
  "Variable",
]


class Variable(object):
  """
  Surrogate object that represents the contents of a
  `tensorflow.core.framework.variable_pb2.VariableDef`
  protocol buffer, which in turn represents a TensorFlow variable.

  Note that the `VariableDef` protobuf is not precisely a public API,
  but it's the closest thing that TensorFlow has to one. Also, you can't
  serialize a general graph in a meaningful way without serializing variables.

  TensorFlow variables are composed of multiple ops and tensors internally.
  TensorFlow's Python API has a class `tf.Variable` that tracks these
  objects. This class tracks a similar set of pointers in protobuf land.
  """
  def __init__(self,
               g # type: graph.Graph
               ):
    """
    Do not call this constructor directly.

    This constructor should only be called from `Graph.add_variable()`.

    Args:
      g: gde.Graph object representing the containing graph
    """
    if g.has_passthrough_saver:
      # The internals of a tf.Saver are opaque to us.
      raise ValueError("Attempted to add a variable to Graph '{}', which has "
                       "an immutable serialized tf.Saver "
                       "object.".format(g.name))
    self._graph = g
    self._collection_names = set()  # Set[str]

    # Core fields are modeled after those of VariableDef.
    self._variable_name = None  # str
    self._initial_value_name = None  # str
    self._initializer_name = None  # str
    self._snapshot_name = None  # str
    self._trainable = None  # bool

  def __str__(self):
    return "Var[{}]".format(self.name)

  def __repr__(self):
    return "Var[name={}, init={}, val={}, " \
           "snap={}, t={}]".format(self.name, self._initializer_name,
                                   self._initial_value_name,
                                   self._snapshot_name,
                                   self._trainable)

  def is_same_variable(self,
                       other # type: Variable
                       ):
    """
    Returns true if is variable and `other` are the same, ignoring graph and
    collection information.
    """
    if self.name != other.name:
      return False
    elif self.initial_value_name != other.initial_value_name:
      return False
    elif self.initializer_name != other.initializer_name:
      return False
    elif self.snapshot_name != other.snapshot_name:
      return False
    elif self.trainable != other.trainable:
      return False
    else:
      return True

  def from_proto(self,
                 variable_def, # type: Union[variable_pb2.VariableDef, bytes]
                 validate=True, # type: bool
                 allow_duplicates=False # type: bool
                 ):
    """
    Populate the fields of this object from a serialized TensorFlow variable.

    variable_def: Protocol buffer representation of a TensorFlow variable. In a
      serialized graph, you will find these VariableDef protocol buffer
      messages stuffed into the `bytes_list` field of a `CollectionDef` proto
      inside a `MetaGraphDef` message. Otherwise you can create a
      `VariableDef` proto by calling `tf.Variable.to_proto()`.
      May be serialized as a `bytes` object.
    validate: True to validate any names used here. False to skip
      validation (e.g because you are creating the variable before creating
      the nodes it references).
      The variable name is checked for duplicates regardless of whether this
      flag is set to True.
    allow_duplicate: Don't complain if the graph contains a variable of the
       same name, provided that the two variables are equal.
    Raises:
      NameError if a variable with the indicated name already exists,
      ValueError if another validation fails
    """
    if isinstance(variable_def, bytes):
      variable_def = variable_pb2.VariableDef.FromString(variable_def)
    self._variable_name = variable_def.variable_name
    self._initial_value_name = variable_def.initial_value_name
    self._initializer_name = variable_def.initializer_name
    self._snapshot_name = variable_def.snapshot_name
    self._trainable = variable_def.trainable
    # TODO(frreiss): Figure out what to do with the is_resource field
    # TODO(frreiss): Figure out what to do with the save_slice_info_def field
    if validate:
      self.validate(allow_duplicates)

  def to_proto(self):
    """
    Inverse of `from_proto()` method.

    Returns a `VariableDef` protocol buffer message that represents this
    variable.
    """
    ret = variable_pb2.VariableDef()
    ret.variable_name = self._variable_name
    ret.initial_value_name = self._initial_value_name
    ret.initializer_name = self._initializer_name
    ret.snapshot_name = self._snapshot_name
    ret.trainable = self._trainable
    return ret

  def validate(self,
               allow_duplicate=False # type: bool
               ):
    """
    Verify that all the names this variable references are valid in the
    parent graph and that no conflicting variables exist.

    Args:
      allow_duplicate: Don't complain if the graph contains a variable of the
        same name, provided that the two variables are equal.
    """
    if self._variable_name in self.graph.variable_names:
      other_var = self.graph.get_variable_by_name(self._variable_name)
      if other_var is not self:
        if not self.is_same_variable(other_var):
          raise ValueError("Existing '{}' in graph conflicts with this one "
                           "({} != {})".format(self._variable_name, repr(self),
                                               repr(other_var)))
        elif not allow_duplicate:
          raise ValueError("Graph already has a variable called '{}'".format(
            self._variable_name))
    # self._initializer_name should reference a node. Other names should
    # reference tensors.
    if not self.graph.contains_node(self._initializer_name):
      raise ValueError("Initializer name '{}' does not correspond to any "
                       "node in graph".format(self._initializer_name))
    _ = self.graph.get_tensor_by_name(self._initial_value_name,
                                      "Invalid initial value name '{}': {}")
    _ = self.graph.get_tensor_by_name(self._snapshot_name,
                                      "Invalid snapshot name '{}': {}")

  def to_proto(self):
    # type: () -> variable_pb2.VariableDef
    """
    Convert this object into its equivalent TensorFlow protocol buffer
    message.

    Returns a `VariableDef` protobuf equivalent to this object.
    """
    ret = variable_pb2.VariableDef()
    ret.variable_name = self.name
    ret.initial_value_name = self.initial_value_name
    ret.initializer_name = self.initializer_name
    ret.snapshot_name = self.snapshot_name
    ret.trainable = self.trainable
    # TODO(frreiss): Figure out what to do with the is_resource field
    # TODO(frreiss): Figure out what to do with the save_slice_info_def field
    return ret


  @property
  def graph(self):
    return self._graph

  @property
  def name(self):
    return self._variable_name

  @name.setter
  def name(self,
           val # type: str
           ):
    # TODO(frreiss): Should we update the graph here?
    self._variable_name = val

  @property
  def initial_value_name(self):
    return self._initial_value_name

  @property
  def initializer_name(self):
    return self._initializer_name

  @property
  def snapshot_name(self):
    return self._snapshot_name

  @property
  def trainable(self):
    return self._trainable

  @property
  def collection_names(self):
    # type: () -> AbstractSet[str]
    """
    Returns the names of all collections this variable is a member of in the
    parent graph.
    """
    return frozenset(self._collection_names)

  def add_to_collection(self,
                        collection_name # type: str
                        ):
    """
    Add the variable to the indicated collection.
    """
    if collection_name not in self._collection_names:
      self._collection_names.add(collection_name)
      # Invalidate any information the parent graph may have cached about
      # collections.
      self._graph.increment_version_counter()



###############################################################################
# Functions below this line are private to this file.

