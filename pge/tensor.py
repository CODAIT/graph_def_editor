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

__all__ = [
  "Tensor",
]


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
