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

"""
rewrite.py

Graph rewrites that ship with the GraphDef Editor.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Tuple, Dict, FrozenSet, Iterable, Union

from graph_def_editor import graph, node, util, tensor, variable


__all__ = [
  "change_batch_size",
]


def change_batch_size(g: graph.Graph,
                      new_size: int,
                      inputs: Iterable[Union[node.Node, tensor.Tensor]]):
  """
  Change the batch size of a model.

  Runs size inference over the graph to propagate the new batch size
  throughout the graph.

  Modifies the graph in place. If the rewrite fails, the graph may be left
  in an inconsistent state.

  Args:
    g: The graph on which to modify the batch size. Modified in place.
    new_size: New batch size to apply on the input(s) to the graph.
      Can be `None` to indicate dynamic batch size.
    inputs: Placeholder nodes that are the input to the graph, either
      the `Node`s themselves or as their output `Tensor`s
  """
  input_nodes = [i.node if isinstance(i, tensor.Tensor) else i
                 for i in inputs]

  # Basic sanity checks
  for n in input_nodes:
    if n.op_type != "Placeholder":
      raise ValueError("Input node {} is not a Placeholder".format(n))
    if n.graph is not g:
      raise ValueError("Input node {} is not in graph {}".format(n, g))

  # Update input nodes
  for n in input_nodes:
    orig_shape = n.get_attr("shape")
    new_dims = [d for d in orig_shape.dims]
    new_dims[0] = new_size
    n.replace_attr("shape", tf.TensorShape(new_dims))

  # Propagate new batch size throughout graph
  g.infer_shapes_and_dtypes()





