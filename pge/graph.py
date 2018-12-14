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
from typing import Tuple

from pge import node

__all__ = [
  "Graph",
]


class Graph(object):
  """
  Mutable wrapper for tf.GraphDef.

  Stores a reference to the original immutable tf.GraphDef and information about
  changes made to the graph.

  Also stores collection information that would be serialized in MetaGraphDef

  Summary of internal data structures:
  * _immutable_nodes: The original immutable GraphDef protobuf
  * _deleted_nodes: Tombstones for nodes removed from the original graph,
                    stored as a set of strings. String == name of removed node
  * _added_nodes: New nodes added to the graph, stored as a dictionary. Key
                  is name.
  * _version: Counter that increments every time the graph is modified
  * _collections: Map from collection name to collection contents for all
                  collections
  """

  def __init__(self, graph: tf.GraphDef = None):
    """
    Wrap a tf.GraphDef protocol buffer in a Graph object.

    Args:
      graph: a tf.Graph or tf.GraphDef protobuf that represents a
        TensorFlow operator graph. If set to None, generate an empty
        tf.GraphDef
    """
    if graph is None:
      graph_def = tf.GraphDef()
    elif isinstance(graph, tf.GraphDef):
      graph_def = graph
    elif isinstance(graph, tf.Graph):
      graph_def = graph.as_graph_def()
    else:
      raise TypeError("Graph is of type {}. Expected a tf.Graph or GraphDef "
                      "proto".format(type(graph)))
    self._graph_def = graph_def
    self._next_id = 1
    output_map = _decode_graph(graph_def)
    self._immutable_nodes = [node.ImmutableNode(self, self._get_next_id(),n,
                                                output_map[n.name])
                             for n in graph_def.node]
    self._deleted_nodes = set()
    self._added_nodes = {}
    self._version = 0
    self._collections = {}

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
    ret = node.MutableNode(self, self._get_next_id(), name, op)
    self._added_nodes[name] = ret
    self._increment_version_counter()
    return ret

  @property
  def nodes(self) -> Tuple[node.Node]:
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

################################################################################
# Stuff below this line is private to this file.

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
