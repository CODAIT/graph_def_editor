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
"""Objects for representing entire graphs undergoing rewrite operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
from distutils import dir_util
import os
from six import string_types
import tensorflow as tf
import sys
if sys.version >= '3':
  from typing import Tuple, Dict, FrozenSet, Iterable, Union, Set, Any

from graph_def_editor import node, util, tensor, variable

# TODO: Move this protobuf into this project so we don't depend on
#  tf.core.framework
from tensorflow.core.protobuf import saved_model_pb2, meta_graph_pb2


__all__ = [
  "Graph",
  "SaverInfo",
  "SignatureInfo",
  "GraphVisitor",
  "saved_model_to_graph",
]

# Special attribute in which TensorFlow stores frame names for while loops (
# see node_to_frame_name() for more information
_FRAME_NAME_ATTR = "frame_name"


class GraphVisitor(object):
  """
  Visitor callback for various graph traversals
  """
  def visit_node(self, n):
    # type: (node.Node) -> None
    raise NotImplementedError()


class SaverInfo(object):
  """
  Object to encapsulate information about a `tf.train.Saver` object that can
  reconstitute the variable values for this graph.
  """
  def __init__(self, path, saver_def):
    # type: (str, tf.train.SaverDef) -> None
    """
    Args:
      path: Path to the location of serialized variable information on disk
      saver_def: Serialized version of `tf.train.Saver` object
    """
    self.path = path
    self.saver_def = saver_def


class SignatureInfo(object):
  """
  Object that encapsulates information about entry points to the graph,
  AKA signatures.
  """
  def __init__(self):
    self._signature_defs = {}  # type: Dict[str, meta_graph_pb2.SignatureDef]

  def add_signature_def(self, name, signature_def):
    # type: (str, meta_graph_pb2.SignatureDef) -> None
    """
    Add a signature to the set of entry points.

    Args:
      name: Name for the entry point
      signature_def: Definition of the entry point; specifies input and
        output nodes and maps them to input and output names
    """
    if name in self._signature_defs:
      raise ValueError("Already have a signature with name '{}'".format(name))
    self._signature_defs[name] = signature_def

  @property
  def signature_defs(self):
    # type: () -> Dict[str, Any]
    return self._signature_defs


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

  def __init__(
          self,
          g = None, # type: Union[tf.Graph, tf.GraphDef]
          name = None, # type: str
          collections = None, # type: Dict[str, meta_graph_pb2.CollectionDef]
          saver_info = None, # type: SaverInfo
          signature_info = None # type: SignatureInfo
          ):
    """
    Wrap a tf.GraphDef protocol buffer in a Graph object.

    Args:
      g: a tf.Graph or tf.GraphDef protobuf that represents a
        TensorFlow graph. If set to None, generate an empty
        tf.GraphDef
      name: Optional human-readable name for the graph. If not provided,
        the constructor will generate a name.
      collections: Optional iterable of tf.MetaGraphDef.CollectionDefEntry 
        objects containing information about collections in the graph.
        Note that this constructor will pull collection info out of `g` if
        it is a `tf.Graph` and `collections` is `None`.
      saver_info: Optional serialiazed information about the
        `tf.train.Saver` object that can save and restore variables in this
        graph.
      signature_info: Optional semi-serialized information about entry points
        to the graph, AKA signatures
    """
    if g is None:
      graph_def = tf.GraphDef()
    elif isinstance(g, tf.GraphDef):
      graph_def = g
    elif isinstance(g, tf.Graph):
      graph_def = g.as_graph_def()
      if collections is None:
        meta_gd = tf.train.export_meta_graph(graph=g)
        collections = _extract_collection_defs(meta_gd)
    else:
      raise TypeError("Graph is of type {}. Expected a tf.Graph or GraphDef "
                      "proto".format(type(g)))
    if name is None:
      time_str = datetime.datetime.now().isoformat()
      name = "GraphDef Editor Graph created {}".format(time_str)
    if signature_info is None:
      signature_info = SignatureInfo()
    elif not isinstance(signature_info, SignatureInfo):
      raise ValueError("signature_info argument must be a SignatureInfo object")

    # Populate fields of object
    self._name = name  # str
    self._version = 0  # Must happen first; other init code needs self._version
    self._frozen = False  # bool
    self._graph_def = graph_def  # tf.GraphDef
    self._next_id = 1  # int
    output_map = _decode_graph(graph_def)
    self._node_name_to_node = {}  # Dict[str, node.Node]; key is node name
    self._node_to_frame_names = None
    self._frame_name_to_nodes = None
    self._head_name_to_coloc_group = None  # Dict[str, FrozenList[str]]
    self._variable_name_to_variable = {}  # Dict[str, Variable]
    self._collection_name_to_type = None  # Dict[str, str], generated on demand
    self._passthrough_collections = {}  # Dict[str, List[CollectionDef]]
    self._passthrough_saver = None
    self._passthrough_versions = graph_def.versions  # tf.VersionDef

    # Load nodes in three passes because the g may contain cycles.
    for node_def in graph_def.node:
      self.add_node_from_node_def(node_def, set_inputs=False)
    for node_def in graph_def.node:
        self[node_def.name].set_outputs_from_pairs(output_map[node_def.name])
    for node_def in graph_def.node:
      self[node_def.name].set_inputs_from_strings(node_def.input,
                                                  set_control_inputs=True)
    # Collections reference nodes and variables
    if collections is not None:
      for k, c in collections.items():
        self.add_collection_from_collection_def(k, c)

    # Presence of a passthrough saver prevents adding additional variables,
    # so load after variables are constituted (i.e. from collections)
    self._passthrough_saver = saver_info
    self._signatures = signature_info

  @property
  def name(self):
    """
    Returns human-readable name for this graph. This name may not be unique
    across graphs.
    """
    return self._name

  @property
  def has_passthrough_saver(self):
    return self._passthrough_saver is not None

  def add_node_from_node_def(self, node_def, set_inputs = False):
    # type: (tf.NodeDef, bool) -> node.Node
    """
    Unpack a `tf.NodeDef` protobuf into a mutable `Node` object.'

    Does NOT set the outputs of the node.

    Args:
      g: Graph in which the node will be created
      node_def: Fully-populated NodeDef proto; all fields, including inputs,
        will be used.
      set_inputs: Optional. If True, also populate the data and control inputs
        of the returned Node. This operation will only work if the targets of
        those inputs are already present in the graph.
    """
    ret = self.add_node(name=node_def.name, op_name=node_def.op)
    ret.device = node_def.device
    for key in node_def.attr:
      ret.add_attr(key, util.attr_value_to_python_type(node_def.attr[key]))
    if set_inputs:
      ret.set_inputs_from_strings(node_def.input, set_control_inputs=True)
    return ret

  def add_collection_from_collection_def(
          self,
          collection_name,
          collection_def,
          validate_name = True):
    # type: (str, meta_graph_pb2.CollectionDef, bool) -> None
    """
    Unpack a `tf.MetaGraphDef.CollectionDefEntry` of serialized variables 
    into a collection of variables in this graph. The collection must not exist. 
    Variables that do not already exist will be created.

    Note that this method is intended to be used to bulk-load a collection.
    To add individual items to a collection one-by-one, call the
    `add_to_collection` methods of `Node`, etc., objects.
    
    Args:
      collection_name: Name of collection
      collection_def: Serialized information about the collection
      validate_name: Verify that a collection by this name doesn't already
        exist. Set this argument to False to avoid O(n^2) behavior when
        bulk-loading known-good collection metadata.
    """
    if validate_name and collection_name in self.get_all_collection_keys():
      raise ValueError("Collection '{}' already exists".format(collection_name))
    # The collection is stored in exactly one of five different formats.
    if collection_def.HasField("node_list"):
      for node_name in collection_def.node_list.value:
        # Check if node name is a Tensor type
        if node_name.rfind(':') > -1:
          n = self.get_tensor_by_name(node_name)
        else:
          n = self.get_node_by_name(node_name)
        n.add_to_collection(collection_name)
    elif collection_def.HasField("bytes_list"):
      for serialized_var in collection_def.bytes_list.value:
        var = self.add_variable_from_variable_def(serialized_var,
                                                  skip_if_present=True)
        var.add_to_collection(collection_name)
    elif (collection_def.HasField("int64_list")
          or collection_def.HasField("float_list")
          or collection_def.HasField("any_list")):
      self._passthrough_collections[collection_name] = collection_def
      if self._collection_name_to_type is not None:
        self._collection_name_to_type[collection_name] = "passthrough"
    else:
      raise ValueError("Unknown collection with name: {}".format(
        collection_name))

  def __getitem__(self, name):
    # type: (str) -> Union[tensor.Tensor, 'node.Node']
    """
    Convenience method to retrieve a node or tensor of the graph by name

    Args:
      name: Name of the node or tensor to return. Case-sensitive.

    Returns the named item as a `gde.Node` or `gde.Tensor` object. If there
    is a conflict between node and tensor names, node names win.
    """
    if not isinstance(name, string_types):
      raise TypeError("name must be a string; got type {}".format(type(name)))

    if self.contains_node(name):
      return self._node_name_to_node[name]
    elif self.contains_tensor(name):
      return self.get_tensor_by_name(name)
    else:
      raise ValueError("No node or tensor '{}' found in graph".format(name))

  def get_node_by_name(self, name):
    # type: (str) -> node.Node
    """
    Retrieve a node in the graph by name.

    Args:
      name: Name of the node. Case-sensitive.

    Returns the indicated node as a `gde.Node` object.
    """
    if self.contains_node(name):
      return self._node_name_to_node[name]
    else:
      raise ValueError("No node '{}' found in graph".format(name))

  def contains_node(self, name):
    # type: (str) -> bool
    """
    Returns true if the graph has a node by the indicated name. Exact string
    match.
    """
    if not isinstance(name, string_types):
      raise ValueError("Node name argument is not a string, but is of type "
                       "{}".format(type(name)))
    return name in self._node_name_to_node.keys()

  def add_node(self,
               name, # type: str
               op_name, # type: str
               uniquify_name = False # type: bool
               ):
    # type: (...) -> node.Node
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

  def add_node_from_node_def(self,
                             node_def, # type: tf.NodeDef
                             set_inputs = False, # type: bool
                             set_control_inputs = False # type: bool
                             ):
    # type: (...) -> node.Node
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

  def remove_node_by_name(self, name, check_for_refs = True):
    # type: (str, str) -> None
    """
    Removes the indicated node from this graph and from any collections in
    this graph.

    The caller is responsible for removing all links to the indicated node
    prior to making this call.

    Args:
      name: name of the node to remove
      check_for_refs: Optional. If True, raise an exception if there are any
        other nodes in the graph that reference this node. If False, allow
        removal of nodes with outstanding references to them. In the latter
        case, the caller is responsible for cleaning up the graph afterwards.
    """
    n = self.get_node_by_name(name)
    if check_for_refs:
      for t in n.outputs:
        if len(t.consumers()) > 0:
          raise ValueError("Removing node '{}' would leave dangling "
                           "references from nodes {} to tensor '{}'"
                           "".format(name, [c.name for c in t.consumers()],
                                     t.name))
    # noinspection PyProtectedMember
    n._remove_from_graph()
    del self._node_name_to_node[name]
    self.increment_version_counter()
    # Don't need to update collection info because collection membership is
    # stored in the node.
    # Don't need to update consumers of tensors because that information is
    # calculated dynamically by iterating over nodes.

  def rename_node(self, old_name, new_name):
    # type: (str, str) -> None
    """
    Change the name of a node in the graph.

    Args:
      old_name: Name of an existing node
      new_name: New name for the node in question. Must not currently be in use.
    """
    if self.contains_node(new_name):
      raise ValueError("Graph already has a node under name '{}'".format(
        new_name))
    n = self.get_node_by_name(old_name)
    # noinspection PyProtectedMember
    n._change_name(new_name)
    del self._node_name_to_node[old_name]
    self._node_name_to_node[new_name] = n
    self.increment_version_counter()

  def add_variable(self, name):
    # type: (str) -> variable.Variable
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
                                     skip_if_present = False):
    # type: (Any, bool) -> None
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

  def get_variable_by_name(self, name):
    # type: (str) -> variable.Variable
    """
    Fetch a variable by its variable name.

    Args:
      name: Name of a variable in this graph.

    Returns the variable associated with the name. Raises an exception if
    there is no variable with the indicated name.
    """
    return self._variable_name_to_variable[name]

  def _name_in_use(self, name):
    # type: (str) -> bool
    """Check whether a name is in use, using the same collision semantics as
    TensorFlow: Exact lowercase string match.

    Args:
      name: Name of a potential node in the graph.

    Returns True if the indicated name is currently in use, ignoring case.
    """
    return name.lower() in [k.lower() for k in self._node_name_to_node.keys()]

  def unique_name(self, name):
    # type: (str) -> str
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
  def node_names(self):
    # type: () -> Iterable[node.Node]
    return self._node_name_to_node.keys()

  @property
  def nodes(self):
    # type: () -> Tuple[node.Node]
    """
    Returns:
      A list of all nodes, both immutable and mutable, present in the graph
      after the edits that this object is buffering.
    """
    return tuple(self._node_name_to_node.values())

  @property
  def tensors(self):
    # type: () -> List[tensor.Tensor]
    """
    Return a list of all the tensors which are input or output of an op in
    the graph.
    """
    ts = []
    for op in self.nodes:
      ts += op.outputs
    return ts

  def contains_tensor(self, tensor_name):
    # type: (str) -> bool
    """
    Returns true if the graph has a tensor by the indicated name. Exact string
    match.

    Args:
      tensor_name: TensorFlow-format name ('node name:input num', or 'node
        name' as shorthand for 'node name:0')

    Raises ValueError if the tensor name is not properly formatted.
    """
    error_msg = "Invalid tensor name '{}': {}"
    node_name, output_ix = _decode_tensor_name(tensor_name, error_msg)
    if node_name not in self._node_name_to_node:
      return False
    else:
      n = self[node_name]
      if output_ix >= len(n.outputs):
        return False
      else:
        return True

  def get_tensor_by_name(self, tensor_name, error_msg = None):
    # type: (str, str) -> tensor.Tensor
    """
    Retrieve a tensor by human-readable name.

    Args:
      tensor_name: TensorFlow-format name ('node name:input num', or 'node
        name' as shorthand for 'node name:0')
      error_msg: Optional format string for raising errors. Must be able to
        serve as an input to `str.format()` with two arguments: tensor name
        string and reason for failure.

    Returns: gde.Tensor object corresponding to the indicated tensor.

    Raises ValueError if the name is invalid or references a tensor that does
    not exist.
    """
    if error_msg is None:
      error_msg = "Invalid tensor name '{}': {}"
    node_name, output_ix = _decode_tensor_name(tensor_name, error_msg)
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
    return n.output(output_ix)

  def to_graph_def(self, add_shapes = True):
    # type: (bool) -> tf.GraphDef
    """
    Args:
      add_shapes: If True, add the special "_output_shapes" attribute with
        output shape information from this Node's output metadata.

    Returns the `tf.GraphDef` serialization of this graph in its current
    form.
    """
    ret = tf.GraphDef()
    ret.versions.CopyFrom(self._passthrough_versions)
    for op in self.nodes:
      op.to_node_def(ret.node.add(), add_shapes)
    return ret

  def to_tf_graph(self):
    # type: () -> tf.Graph
    """
    Converts this graph into a new TensorFlow `Graph`. Also takes care of
    variables.

    Returns a fresh `tf.Graph` containing all the nodes and variables that
    this object represents.
    """
    ret = tf.Graph()
    with ret.as_default():
      tf.import_graph_def(self.to_graph_def(), name="")
      util.load_variables_to_tf_graph(self)
    return ret

  def to_saved_model(self, saved_model_path, tags = None):
    # type: (str, Iterable[str]) -> saved_model_pb2.SavedModel
    """
    Writes this graph out as a TensorFlow SavedModel on disk.

    Args:
      saved_model_path: Location where the root directory of the SavedModel
        should reside.
      tags: What tag strings should be associated with the MetaGraph that this
        method puts inside the SavedModel. If None, use the
        tag `tf.saved_model.tag_constants.SERVING`

    Returns the SavedModel protocol buffer message that it wrote to the
    specified location.
    """
    if tags is None:
      tags = [tf.saved_model.tag_constants.SERVING]
    if os.path.exists(saved_model_path):
      raise ValueError("Output path '{}' already exists".format(
        saved_model_path))
    if not os.path.exists(os.path.dirname(saved_model_path)):
      raise ValueError("Parent directory '{}' of output dir '{}' does not "
                       "exist".format(os.path.dirname(saved_model_path),
                                      saved_model_path))
    os.mkdir(saved_model_path)

    # Core part of the SavedModel is a protocol buffers file containing a
    # SavedModel protocol buffer message.
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
    # core/protobuf/saved_model.proto
    saved_model = saved_model_pb2.SavedModel()
    saved_model.saved_model_schema_version = 1

    # Inside the SavedModel protobuf is a list of MetaGraphDef protobufs. In
    # this case there is only one.
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
    # core/protobuf/meta_graph.proto
    meta_graph = saved_model.meta_graphs.add()

    # The MetaGraphDef message contains a nested header called a MetaInfoDef.
    # The first field of the MetaInfoDef is called "meta_graph_version".
    # This field does not actually hold the version of the MetaGraph. Instead
    # it holds an arbitrary string that can be whatever you want.
    meta_info_def = tf.MetaGraphDef.MetaInfoDef()
    meta_info_def.meta_graph_version = self.name

    # The second field, "stripped_op_list" holds "A copy fo the OpDefs used by
    # the producer of this graph_def". According to the docs for
    # tf.import_graph_def, this field is deprecated. This field does not
    # appear to have ever accomplished anything useful.
    # TensorFlow fills this field with a deluge of OpDef information. We leave
    # this field out.

    # The third field, "any_info", provides a place for holding additional
    # arbitrary information. We also leave this field out.

    # The fourth field holds the string tags for this MetaGraph
    meta_info_def.tags.extend(tags)

    # The fifth and sixth fields hold TensorFlow version information.
    # We punt here and populate these fields with the version info from
    # the current Python session's copy of TensorFlow.
    meta_info_def.tensorflow_version = tf.VERSION
    meta_info_def.tensorflow_git_version = tf.GIT_VERSION

    # The final field, "stripped_default_attrs", is "A flag to denote whether
    # default-valued attrs have been stripped from the nodes in this graph_def"
    # The TensorFlow authors appear to have added this field in the hopes
    # that future versions of the system might be able to use it for forwards
    # compatibility. No code in TensorFlow currently reads this attribute. We
    # set it to False.
    meta_info_def.stripped_default_attrs = False

    meta_graph.meta_info_def.CopyFrom(meta_info_def)

    # After the meta_info_def comes a GraphDef proto holding all the graph
    # nodes that this MetaGraph uses. If an op in the original TensorFlow
    # graph is in multiple MetaGraphs, that op will be stored ONCE PER
    # METAGRAPH under this field. In our case there is exactly one
    # MetaGraph in the SavedModel.
    meta_graph.graph_def.CopyFrom(self.to_graph_def())

    # The next field, "saver_def", holds information about the tf.Saver
    # instance that will be used to reconstitute any variables in the graph
    if self.has_passthrough_saver:
      meta_graph.saver_def.CopyFrom(self._passthrough_saver.saver_def)
      # Copy serialized variables checkpoint wholesale, because the checkpoint
      # format is a black box to us.
      dir_util.copy_tree(self._passthrough_saver.path,
                         _vars_dir_for_saved_model(saved_model_path))
    elif len(self.variable_names) > 0:
      raise NotImplementedError("Can't generate a SaverDef.")
    else:
      # Zero variables, no passthrough SaverDef.
      # For this case, TensorFlow creates an empty variables directory and
      # doesn't set the "saver_def" field. We emulate this behavior.
      os.mkdir(_vars_dir_for_saved_model(saved_model_path))

    # The next field, "collection_def", holds serialized information about all
    # collections in the MetaGraph.
    if self._collection_name_to_type is None:
      self._build_collection_name_to_type()
    for coll_name, coll_type in self._collection_name_to_type.items():
      if coll_type == "passthrough":
        meta_graph.collection_def[coll_name] = self._passthrough_collections[
          coll_name]
      elif coll_type == "variable":
        vars_list = self.get_collection_by_name(coll_name)
        serialized_vars = [v.to_proto().SerializeToString() for v in vars_list]
        meta_graph.collection_def[coll_name].bytes_list.value.extend(
          serialized_vars)
      elif coll_type == "node":
        nodes_list = self.get_collection_by_name(coll_name)
        meta_graph.collection_def[coll_name].node_list.value.extend(
          [n.name for n in nodes_list])
      else:
        raise ValueError("Unknown collection type '{}'".format(coll_type))

    # The next field, "signature_def", contains information about
    # input/output signatures that this MetaGraph  supports.
    for sig_name, sig_def in self.signatures.items():
      meta_graph.signature_def[sig_name].CopyFrom(sig_def)

    # The final field, asset_file_def, stores information about additional
    # assets that are packaged along with the graph in the SavedModel's
    # "assets" directory. Fow now we leave this field empty.
    # TODO(frreiss): Represent assets as a field in the Graph class and
    #  serialize them here.

    # At this point, we have created the root directory for the SavedModel,
    # as well as the checkpoints directory. The only thing left to write is
    # the SavedModel protobuf itself.
    with open(saved_model_path + "/saved_model.pb", "wb") as f:
      f.write(saved_model.SerializeToString())
    return saved_model

  @property
  def version(self):
    # type: () -> int
    """
    Returns a counter that goes up every time this graph is changed.
    """
    return self._version

  @property
  def frozen(self):
    # type: () -> bool
    """
    True if the graph is configured to raise an exception on any structural
    modification.
    """
    return self._frozen

  @frozen.setter
  def frozen(self, value):
    # type: (bool) -> None
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
    self._collection_name_to_type = None

  def get_collection_by_name(self, name):
    # type: (str) -> Iterable[Any]
    """Fetch the contents of a collection, similarly to the method in
    `tf.Graph` by the same name.

    Args:
      name: Name of collection to fetch

    Returns:
      The values in the collection. Currently any type is allowed in these
      values, following the conventions of the TensorFlow APIs.
    """
    if self._collection_name_to_type is None:
      self._build_collection_name_to_type()
    if name not in self._collection_name_to_type:
      raise ValueError("No collection with name '{}'".format(name))
    coll_type = self._collection_name_to_type[name]
    if coll_type == "passthrough":
      return self._passthrough_collections[name]
    elif coll_type == "variable":
      ret = []
      for v_name in self.variable_names:
        v = self.get_variable_by_name(v_name)
        if name in v.collection_names:
          ret.append(v)
      return ret
    elif coll_type == "node":
      ret = []
      for n in self.nodes:
        if name in n.collection_names:
          ret.append(n)
      for t in self.tensors:
        if name in t.collection_names:
          ret.append(t)
      return ret
    else:
      raise ValueError("Unknown collection type '{}'".format(coll_type))

  def _build_collection_name_to_type(self):
    # type: () -> None
    self._collection_name_to_type = {}
    passthrough_collection_names = set(self._passthrough_collections.keys())
    variable_collection_names = set()
    node_collection_names = set()
    for var_name in self.variable_names:
      v = self.get_variable_by_name(var_name)
      for name in v.collection_names:
        variable_collection_names.add(name)
    for n in self.nodes:
      for name in n.collection_names:
        node_collection_names.add(name)
    node_collection_names_tensors = set()
    for t in self.tensors:
      for name in t.collection_names:
        node_collection_names_tensors.add(name)
    if not node_collection_names_tensors.intersection(node_collection_names):
      node_collection_names.update(node_collection_names_tensors)
    else:
      raise TypeError("Node collections cannot be Nodes and Tensors for: "
                      "{}".format(name))

    def _add(names, type_name):
      for coll_name in names:
        if coll_name in self._collection_name_to_type:
          raise ValueError((
            _duplicate_collection_error_str(coll_name,
                                            passthrough_collection_names,
                                            variable_collection_names,
                                            node_collection_names)))
        self._collection_name_to_type[coll_name] = type_name
    _add(passthrough_collection_names, "passthrough")
    _add(variable_collection_names, "variable")
    _add(node_collection_names, "node")

  def get_all_collection_keys(self):
    # type: () -> Iterable[str]
    """Returns the keys associated with all collections stored in this object"""
    if self._collection_name_to_type is None:
      self._build_collection_name_to_type()
    return self._collection_name_to_type.keys()

  def _get_next_id(self):
    # type: () -> int
    """Generates and returns a unique integer ID *within this graph*."""
    ret = self._next_id
    self._next_id = ret + 1
    return ret

  def node_to_frame_names(self, n):
    # type: (node.Node) -> Tuple[str]
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
      TODO: Fix this documentation
      Dictionary mapping from nodes in this graph to the names of
      one or
      more nested frames that will be active when reaching this node.
      The returned value is only valid until this graph is modified, either
      by modifying the link structure of the graph or by changing the
      "frame_name" attribute of an Enter node. Nodes that are not nested
      inside any while loops are mapped to None.
    """
    if self._node_to_frame_names is None:
      self._generate_node_to_frame_name()
    return self._node_to_frame_names[n]

  def frame_name_to_nodes(self, frame_name):
    # type: (str) -> Tuple[node.Node]
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

  def get_frame_names(self):
    # type: () -> Tuple[str]
    """
    Returns:
      Tuple of all the unique names of frames that occur in this graph.
    """
    if self._node_to_frame_names is None:
      self._generate_node_to_frame_name()
    return self._frame_name_to_nodes.keys()

  def breadth_first_visitor(self,
                            visitor, # type: GraphVisitor
                            starting_nodes = None # type: Iterable[node.Node]
                            ):
    # type: (...) -> None
    """
    Visit all nodes reachable from a starting set in the order of a
    breadth-first traversal. Invokes a callback at each node visited.

    Args:
      visitor: Possibly-stateful callback to be invoked on each node reached
      starting_nodes: Optional list of starting nodes. If this set is not
         provided, this method will use all nodes with zero inputs as the
         starting set. Search will visit these nodes first, then visit their
         children in order by parent node.
    """
    if starting_nodes is None:
      # Start with all of the nodes in the graph that have no inputs.
      # The maintainers of the TensorFlow scheduler like to call these nodes
      # "root nodes".
      starting_nodes = [n for n in self.nodes if 0 == len(n.inputs)]

    # Use a Python list as a node queue for the breadth-first search.
    queue = list(starting_nodes)
    enqueued_nodes = set(queue)

    while len(queue) > 0:
      cur_node = queue.pop(0)
      visitor.visit_node(cur_node)

      # Prepare for next stage of search
      for out_tensor in cur_node.outputs:
        for out_node in out_tensor.consumers():
          if out_node not in enqueued_nodes:
            queue.append(out_node)
            enqueued_nodes.add(out_node)

  def infer_shapes_and_dtypes(self,
                              starting_nodes = None # type: Iterable[node.Node]
                              ):
    # type: (...) -> None
    """
    Visit all nodes reachable from a starting set in the order of a
    breadth-first traversal, invoking shape and type inference.

    Args:
      starting_nodes: Optional list of starting nodes. If this set is not
         provided, this method will use all nodes with zero inputs as the
         starting set. Search will visit these nodes first, then visit their
         children in order by parent node.
    """
    class _MyVisitor(GraphVisitor):
      def visit_node(self,
                     cur_node # type: node.Node
                     ):
        cur_node.infer_outputs()
    self.breadth_first_visitor(_MyVisitor(), starting_nodes)

  def _generate_node_to_frame_name(self):
    """
    Regenerate the tables behind the node_to_frame_name and
    frame_name_to_node properties. Performs a breadth-first traversal of the
    graph, duplicating the logic in the function
    ExecutorImpl::BuildControlFlowInfo() in
    tensorflow/core/common_runtime/executor.cc
    """
    class _MyVisitor(GraphVisitor):
      def __init__(self):
        # Maintain a stack of frame names, a la the original code in
        # executor.cc.
        # We use None to represent the root frame.
        self.frame_name_stack = [None]
        self.frame_name_tuple = tuple()  # Immutable version for table

        self.new_node_to_frame_names = {}
        self.new_frame_name_to_nodes = {}

      def visit_node(self,
                     cur_node # type: node.Node
                     ):
        if cur_node.op_type in ["Enter", "RefEnter"]:
          # Entering a while loop. Push a frame name onto the virtual stack
          if _FRAME_NAME_ATTR not in cur_node.get_attr_keys():
            raise ValueError("Node {} is of op type {} but does not have a "
                             "value for its {}"
                             " attribute".format(cur_node.name,
                                                 cur_node.op_type,
                                                 _FRAME_NAME_ATTR))
          self.frame_name_stack.append(cur_node.get_attr(_FRAME_NAME_ATTR))
          self.frame_name_tuple = tuple(self.frame_name_stack)
        elif cur_node.op_type in ["Exit", "RefExit"]:
          self.frame_name_stack.pop(-1)
          self.frame_name_tuple = tuple(self.frame_name_stack)
        # Update tables
        self.new_node_to_frame_names[cur_node] = self.frame_name_tuple
        for f in self.frame_name_stack:
          self.new_frame_name_to_nodes.setdefault(f, []).append(cur_node)

    visitor = _MyVisitor()
    self.breadth_first_visitor(visitor)
    self._node_to_frame_names = visitor.new_node_to_frame_names
    # Reverse mapping was built a dict of lists to avoid O(n^2) behavior.
    # Convert to dict of tuples.
    self._frame_name_to_nodes = {
      k: tuple(v) for k, v in visitor.new_frame_name_to_nodes.items()
    }

  @property
  def colocation_groups(self):
    # type: () -> Dict[str, FrozenSet[node.Node]]
    """
    Generate a table of all groups of nodes that must be on the same device
    according to collocation constrains in the underlying NodeDefs.

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
        k: frozenset(v) for k, v in head_name_to_coloc_group.items()}
    return self._head_name_to_coloc_group

  @property
  def signatures(self):
    # type: () -> Dict[str, meta_graph_pb2.SignatureDef]
    """
    Returns a map from signature name to signature definition. Changes to
    this map will be reflected in this object.
    """
    return self._signatures.signature_defs


def saved_model_to_graph(saved_model_path, # type: str
                         tag = None, # type: str
                         include_saver = True, # type: bool
                         include_signatures = True # type: bool
                         ):
  # type: (...) -> Graph
  """
  Load the contents of a TensorFlow SavedModel into a Graph object.

  Args:
    saved_model_path: Path to the SavedModel's directory on disk
    tag: User-specified tag attached to the MetaGraphDef that should be
      loaded from the SavedModel. If None, verify that there is only one
      MetaGraphDef in the model and load that one.
    include_saver: If True, attach black-box information about the SavedModel's
      serialized `tf.Saver` object in the returned Graph object. Otherwise the
      returned Graph will not contain any serialized variable values, though
      it will contain variable initializers.
    include_signatures: If True, attach signature information from the
      SavedModel to the returned Graph object. Otherwise the returned graph
      will have no signatures.

  Returns: In-memory representation of the contents of the SavedModel as a
  Graph object.
  """
  if not os.path.exists(saved_model_path):
    raise ValueError("SavedModel root directory {} not found".format(
      saved_model_path))
  if not os.path.isdir(saved_model_path):
    raise ValueError("SavedModel root path {} is not a directory".format(
      saved_model_path))

  # By convention, the main protobuf for the SavedModel is in a file called
  # "saved_model.pb"
  protobuf_file = saved_model_path + "/saved_model.pb"
  saved_model = saved_model_pb2.SavedModel()
  with open(protobuf_file, "rb") as f:
    saved_model.ParseFromString(f.read())

  # Drill down to pull out the appropriate MetaGraphDef proto
  if tag is None:
    if len(saved_model.meta_graphs) != 1:
      raise ValueError("No tags specified and there are multiple "
                       "MetaGraphDefs in the SavedModel. Please specify a "
                       "tag to select a specific MetaGraphDef")
    meta_graph = saved_model.meta_graphs[0]
  else:
    matching_ixs = [
      i for i in range(len(saved_model.meta_graphs))
      if tag in saved_model.meta_graphs[i].meta_info_def.tags
    ]
    if len(matching_ixs) == 0:
      raise ValueError("No MetaGraphDef in SavedModel at {} contains tag "
                       "'{}'".format(saved_model_path, tag))
    if len(matching_ixs) > 1:
      raise ValueError("{} different MetaGraphDef in SavedModel at {} "
                       "contain tag '{}'. Please specify a tag that "
                       "uniquely identifies a MetaGraphDef"
                       "".format(len(matching_ixs), saved_model_path, tag))
    meta_graph = saved_model.meta_graphs[matching_ixs[0]]

  # Decompose the MetaGraphDef into the serialized components of the graph
  graph_def = meta_graph.graph_def
  collections = _extract_collection_defs(meta_graph)
  if include_saver and meta_graph.HasField("saver_def"):
    saver_info = SaverInfo(_vars_dir_for_saved_model(saved_model_path),
                           meta_graph.saver_def)
  else:
    saver_info = None
  signature_info = SignatureInfo()
  if include_signatures:
    for key in meta_graph.signature_def:
      signature_info.add_signature_def(key, meta_graph.signature_def[key])

  return Graph(graph_def,
               name=meta_graph.meta_info_def.meta_graph_version,
               collections=collections,
               saver_info=saver_info,
               signature_info=signature_info)

################################################################################
# Stuff below this line is private to this file.


def _decode_graph(graph_def):
  # type: (tf.GraphDef) -> Dict[str, List[Tuple[tf.DType, tf.TensorShape]]]
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


def _extract_collection_defs(meta_graph):
  # type: (tf.MetaGraphDef) -> Dict[str, meta_graph_pb2.CollectionDef]
  collections = {}
  for collection_name in meta_graph.collection_def:
    if type(collection_name) is not str:
      print("Skipping non-string collection name {}".format(collection_name))
      continue
    elif collection_name in (
            "while_context", "cond_context", "savers", "queue_runners"):
      print("Skipping collection {}".format(collection_name))
      # TODO(frreiss): Should we serialize WhileContexts or CondContexts?
      continue
    collections[collection_name] = meta_graph.collection_def[collection_name]

  return collections


def _decode_tensor_name(tensor_name, error_msg):
  # type: (str, str) -> Tuple[str, int]
  """
  Args:
    tensor_name: TensorFlow-format name ('node name:input num', or 'node
      name' as shorthand for 'node name:0')
    error_msg: Format string for raising errors. Must be able to
      serve as an input to `str.format()` with two arguments: tensor name
      string and reason for failure.

  Returns: (node name, output index) tuple identifying the tensor

  Raises ValueError if the name is not properly formatted
  """
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

  return node_name, output_ix


def _duplicate_collection_error_str(
        name, # type: str
        passthrough_collection_names, # type: Set[str]
        variable_collection_names, # type: Set[str]
        node_collection_names # type: Set[str]
  ):
  # type: (...) -> str
  """
  Generate an error string for the case where a collection ends up being of
  multiple types simultaneously.
  """
  types = []
  if name in passthrough_collection_names:
    types.append("passthrough")
  if name in variable_collection_names:
    types.append("variable")
  if name in node_collection_names:
    types.append("node")
  return (
    "Collection name '{}' maps to multiple collection types: "
    "{}".format(name, types))


def _vars_dir_for_saved_model(
        saved_model_path # type: str
  ):
  # type: (str) -> str
  """
  Args:
    saved_model_path: Root directory of a SavedModel on disk

  Returns the location of the directory where the indicated SavedModel will
  store its variables checkpoint.
  """
  return saved_model_path + "/variables"
