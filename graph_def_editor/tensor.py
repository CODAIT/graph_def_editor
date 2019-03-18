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

import tensorflow as tf
import sys
if sys.version >= '3':
  from typing import AbstractSet

__all__ = [
  "Tensor",
]


class Tensor(object):
  """
  Surrogate object that represents an output of a Node. Corresponds roughly to
  a tf.Tensor in the TensorFlow Python API, though serialized TensorFlow graphs
  do not contain any separate objects that represent tensors.
  """
  def __init__(self,
               node,
               index,
               dtype, # type: tf.DType,
               shape # type: tf.shape
               ):
    """
    Args:
      node: gde.Node object that represents the graph node that produces this
        tensor
      index: Output index of this tensor among the outputs of the specified node
      dtype: Data type of the tensor
      shape: Shape of the tensor
    """
    self._node = node
    self._index = index
    self._dtype = dtype
    self._shape = shape
    self._collection_names = set()  # Set[str]

  def __str__(self):
    return "Tensor '{}' (dtype {}, shape {})".format(self.name, self.dtype,
                                                     self.shape)

  def __repr__(self):
    return str(self)

  @property
  def node(self):
    return self._node

  @property
  def op(self):
    """Alias for self.node, for compatibility with code written for
    tf.Tensor"""
    return self.node

  @property
  def value_index(self):
    """
    Emulates the behavior of `tf.Tensor.value_index`

    Returns the output index of this Tensor among the outputs of the parent
    Node."""
    return self._index

  @property
  def dtype(self):
    # type () -> tf.DType:
    return self._dtype

  @dtype.setter
  def dtype(self,
            value # type: tf.DType
            ):
    self._dtype = value

  @property
  def shape(self):
    # type () -> tf.TensorShape
    return self._shape

  @shape.setter
  def shape(self,
            value # type: tf.TensorShape
            ):
    self._shape = value

  @property
  def graph(self):
    """Returns the `gde.Graph` object representing the graph in which the
    node that produces this tensor resides."""
    return self._node.graph

  def consumers(self):
    """Returns the `gde.Node` objects representing the ops that consume the
    tensor that this object represents."""
    # TODO: Maintain a lookup table of graph edges.
    # For now we do linear search for correctness.
    ret = []
    for n in self.graph.nodes:
      if self in n.inputs:
        ret.append(n)
    return ret

  @property
  def name(self):
    """
    Emulates the behavior of `tf.Tensor.name`

    Returns:
      A TensorFlow-like tensor name string in the form "<op>:<output index>"
    """
    return "{}:{}".format(self.node.name, self.value_index)

  @property
  def collection_names(self):
    # type -> AbstractSet[str]
    """
    Returns the names of all collections this tensor is a member of in the
    parent graph.
    """
    return frozenset(self._collection_names)

  def add_to_collection(self,
                        collection_name # type: str
                        ):
    """
    Add the tensor to the indicated collection.
    """
    if collection_name not in self._collection_names:
      self._collection_names.add(collection_name)
      # Invalidate any information the parent graph may have cached about
      # collections.
      self.node._graph.increment_version_counter()

