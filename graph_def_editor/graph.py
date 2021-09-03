# Copyright 2021 Google. All Rights Reserved.
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

from collections import defaultdict
import datetime
import os
from six import string_types
import tensorflow.compat.v1 as tf
import sys
if sys.version >= '3':
  from typing import Tuple, Dict, FrozenSet, Iterable, Union, Set, Any
import queue

from graph_def_editor import base_graph, function_graph, node, util, tensor, variable, subgraph
import graph_def_editor.visualization.graphviz_wrapper as gvw

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
_INPUT_DUMMY_OP_NAME = "__input__"


class GraphVisitor(object):
  """
  Visitor callback for various graph traversals
  """
  def visit_node(self, n):
    # type: (node.Node) -> None
    raise NotImplementedError()

  def __call__(self, n):
    return self.visit_node(n)


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

class Graph(base_graph.BaseGraph):
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
      g=None,  # type: Union[tf.Graph, tf.GraphDef]
      name=None,  # type: str
      collections=None,  # type: Dict[str, meta_graph_pb2.CollectionDef]
      saver_info=None,  # type: SaverInfo
      signature_info=None,  # type: SignatureInfo,
      object_graph_def=None,  # type: saved_object_graph_pb2.SavedObjectGraph
      stripped_op_list=None,  # type: op_def_pb2.OpList
      asset_file_def=None,  # type: meta_graph_pb2.AssetFileDef
  ):
    """Wrap a tf.GraphDef protocol buffer in a Graph object.

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
        to the graph, AKA signatures.
      object_graph_def: Optional SavedObjectGraph for TF2.x.
      stripped_op_list: Optional stripped op list.
      asset_file_def: Optional saved model assets.
    """
    if name is None:
      time_str = datetime.datetime.now().isoformat()
      name = "GraphDef Editor Graph created {}".format(time_str)
    super(Graph, self).__init__(name)
    self._graph = None
    if g is None:
      self._graph_def = tf.GraphDef()
    elif isinstance(g, tf.GraphDef):
      self._graph_def = g
    elif isinstance(g, tf.Graph):
      self._graph_def = g.as_graph_def()
      self._graph = g
      if collections is None:
        meta_gd = tf.train.export_meta_graph(graph=g)
        collections = _extract_collection_defs(meta_gd)
    else:
      raise TypeError("Graph is of type {}. Expected a tf.Graph or GraphDef "
                      "proto".format(type(g)))
    if signature_info is None:
      signature_info = SignatureInfo()
    elif not isinstance(signature_info, SignatureInfo):
      raise ValueError("signature_info argument must be a SignatureInfo object")

    # Caching tf.Graph object, so we won't have to load it again.
    if self._graph is None:
      self._graph = tf.Graph()
      with self._graph.as_default():
        tf.import_graph_def(self._graph_def, name="")

    # Populate fields of object
    self._version = 0  # Must happen first; other init code needs self._version
    self._frozen = False  # bool
    self._next_id = 1  # int
    self._node_name_to_node = {}  # Dict[str, node.Node]; key is node name
    output_map = _decode_graph(self._graph)
    self._node_to_frame_names = None
    self._frame_name_to_nodes = None
    self._head_name_to_coloc_group = None  # Dict[str, FrozenList[str]]
    self._variable_name_to_variable = {}  # Dict[str, Variable]
    self._collection_name_to_type = None  # Dict[str, str], generated on demand
    self._passthrough_collections = {}  # Dict[str, List[CollectionDef]]
    self._passthrough_saver = None
    self._passthrough_versions = self._graph_def.versions  # tf.VERSIONDef
    self._function_graphs = dict()  # Dict[str, gde.FuncGraph], on demand

    # Load nodes in three passes because the g may contain cycles.
    for node_def in self._graph_def.node:
      self.add_node_from_node_def(node_def, set_inputs=False)
    for node_def in self._graph_def.node:
      self[node_def.name].set_outputs_from_pairs(output_map[node_def.name])
    for node_def in self._graph_def.node:
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
    self._object_graph_def = object_graph_def
    self._stripped_op_list = stripped_op_list
    self._asset_file_def = asset_file_def

  @property
  def has_passthrough_saver(self):
    return self._passthrough_saver is not None

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
        if node_name.rfind(":") > -1:
          n = self.get_tensor_by_name(node_name)
        else:
          n = self.get_node_by_name(node_name)
        n.add_to_collection(collection_name)
    elif collection_def.HasField("bytes_list"):
      for serialized_var in collection_def.bytes_list.value:
        var = self.add_variable_from_variable_def(serialized_var,
                                                  skip_if_present=True)
        var.add_to_collection(collection_name)
    elif (collection_def.HasField("int64_list") \
          or collection_def.HasField("float_list") \
          or collection_def.HasField("any_list")):
      self._passthrough_collections[collection_name] = collection_def
      if self._collection_name_to_type is not None:
        self._collection_name_to_type[collection_name] = "passthrough"
    else:
      raise ValueError("Unknown collection with name: {}".format(
          collection_name))

  @property
  def function_names(self):
    # type: () -> Iterable[str]
    return [f.signature.name for f in self._graph_def.library.function]

  def get_function_graph_by_name(self, function_name):
    """
    Retrieve a function by name and wrap it into a function_graph.FunctionGraph.

    Args:
      function_name: Function name.

    Returns: function_graph.FunctionGraph object.

    Raises: ValueError if the function with specified name is not found
      in the graph.
    """
    if function_name not in self.function_names:
      raise ValueError("Function '{}' is not found in graph".format(
          function_name))

    if function_name not in self._function_graphs:
      self._function_graphs[function_name] = function_graph.FunctionGraph(
          name=function_name,
          parent_tf_graph=self._graph,
          parent_graph=self)

    return self._function_graphs[function_name]

  def to_graph_def(self, add_shapes=True):
    # type: (bool) -> tf.compat.v1.GraphDef
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

    # Copy library as is.
    if self._graph_def and self._graph_def.library:
      ret.library.CopyFrom(self._graph_def.library)

    # Update functions in library that were instantiated as function graphs.
    for f_name, f_graph in self._function_graphs.items():
      function_index_to_update = None
      for index in range(0, len(ret.library.function)):
        if ret.library.function[index].signature.name == f_name:
          function_index_to_update = index
          break
      if function_index_to_update is None:
        ValueError("Function '{}' is not found in graph".format(f_name))
      ret.library.function[function_index_to_update].Clear()
      ret.library.function[function_index_to_update].MergeFrom(
          f_graph.to_function_graph_def())
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
    if tf.gfile.Exists(saved_model_path):
      raise ValueError("Output path '{}' already exists".format(
        saved_model_path))
    saved_model_path = saved_model_path.rstrip("/")
    if not tf.gfile.Exists(os.path.dirname(saved_model_path)):
      raise ValueError("Parent directory '{}' of output dir '{}' does not "
                       "exist".format(os.path.dirname(saved_model_path),
                                      saved_model_path))
    tf.gfile.MakeDirs(saved_model_path)

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

    # Passing through stripped_op_list.
    if self._stripped_op_list:
      meta_info_def.stripped_op_list.CopyFrom(self._stripped_op_list)

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
      if not tf.gfile.Exists(_vars_dir_for_saved_model(saved_model_path)):
        tf.gfile.MkDir(_vars_dir_for_saved_model(saved_model_path))
      # Copy serialized variables checkpoint wholesale, because the checkpoint
      # format is a black box to us.
      util.copy_directory(self._passthrough_saver.path,
                          _vars_dir_for_saved_model(saved_model_path),
                          overwrite=True)
    elif len(self.variable_names) > 0:
      raise NotImplementedError("Can't generate a SaverDef.")
    else:
      # Zero variables, no passthrough SaverDef.
      # For this case, TensorFlow creates an empty variables directory and
      # doesn't set the "saver_def" field. We emulate this behavior.
      tf.gfile.MkDir(_vars_dir_for_saved_model(saved_model_path))

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
    # "assets" directory.
    if self._asset_file_def:
      meta_graph.asset_file_def.extend(self._asset_file_def)
      from_assets_path = os.path.join(self._passthrough_saver.path, "..",
                                      "assets")
      if tf.gfile.Exists(from_assets_path):
        to_assets_path = os.path.join(saved_model_path, "assets")

        if not tf.gfile.Exists(to_assets_path):
          tf.gfile.MkDir(to_assets_path)
        util.copy_directory(from_assets_path,
                            to_assets_path,
                            overwrite=True)

    # It should be fine copying object_graph_def, as function signature
    # changes are not supported.
    if self._object_graph_def:
      meta_graph.object_graph_def.CopyFrom(self._object_graph_def)

    # At this point, we have created the root directory for the SavedModel,
    # as well as the checkpoints directory. The only thing left to write is
    # the SavedModel protobuf itself.
    with tf.gfile.Open(saved_model_path + "/saved_model.pb", "wb") as f:
      f.write(saved_model.SerializeToString())
    return saved_model


  def increment_version_counter(self):
    """
    Mark the structure of this graph as "changed" and invalidate any cached
    information about the edges of the graph.
    """
    super(Graph, self).increment_version_counter()
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

  def nodes_iterator(
      self,
      predicate=lambda _: True,  # type: (node.Node) -> bool
      iterate_functions=False  # type: bool
      ):
    # type: (...) -> Iterable[node.Node]
    """
    Returns:
      An iterator over nodes matching predicate in current graph and
      from function graphs if iterate_functions=True.
    """
    for op in self.nodes:
      if predicate(op):
        yield op
    if iterate_functions:
      for function_name in self.function_names:
        for op in self.get_function_graph_by_name(function_name).nodes:
          if predicate(op):
            yield op

  def breadth_first_visitor(
      self,
      visitor,  # type: callable
      starting_nodes=None,  # type: Iterable[node.Node]
      visited_nodes=None,  # type: set
      iterate_functions=False,  # type: bool
      escape_functions=False,  # type: bool
      max_depth=None  # type: int
  ):
    # type: (...) -> None
    """
    Visit all nodes reachable from a starting set in the order of a
    breadth-first traversal (going from node to output edges).
    If visitor gets to a function call, and iterate_functions is True,
    it will iterate all function nodes first and then continue with
    remaining nodes in the graph.
    Invokes a callback at each node visited.

    Args:
      visitor: Possibly-stateful callback to be invoked on each node reached
      starting_nodes: Optional list of starting nodes. If this set is not
          provided, this method will use all nodes with zero inputs as the
          starting set. Search will visit these nodes first, then visit their
          children in order by parent node.
      visited_nodes: Optional set of nodes to skip iterating over.
      iterate_functions: Indicates if we should also go inside functions if one
          is found in the graph.
      escape_functions: If iteration started in a function graph, indicates
          that we should also iterate through all function callers up
          the stack.
      max_depth: Maximum depth to iterate up to. If None, iteration will
          continue until boundary of the graph is reached.
    Returns:
      True if iteration was interrupted by visitor, otherwise False.
    """
    if starting_nodes is None:
      # Start with all of the nodes in the graph that have no inputs.
      # The maintainers of the TensorFlow scheduler like to call these nodes
      # "root nodes".
      starting_nodes = [n for n in self.nodes if not n.inputs]

    if visited_nodes is None:
      visited_nodes = set()

    nodes_queue = queue.Queue()
    function_graph_names_set = set()

    starting_nodes_set = set()
    for n in starting_nodes:
      nodes_queue.put((n, None, max_depth))
      starting_nodes_set.add(n)

    while not nodes_queue.empty():
      (n, input_tensor, depth) = nodes_queue.get()
      if n in visited_nodes:
        continue
      if n.op_type != _INPUT_DUMMY_OP_NAME:
        if visitor(n):
          return True

      if escape_functions and isinstance(n.graph, function_graph.FunctionGraph):
        function_graph_names_set.add(n.graph.name)

      visited_nodes.add(n)
      if iterate_functions and n.op_type in node.PARTITIONED_CALL_OP_TYPES:
        function_name = n.get_attr("f").name
        f_graph = self.get_function_graph_by_name(function_name)

        function_inputs = []
        if input_tensor is not None:
          for input_node in f_graph.nodes:
            if input_node.op_type == _INPUT_DUMMY_OP_NAME and input_node.name in input_tensor.name:
              function_inputs.append(input_node)

        if len(function_inputs) == 0:
          function_inputs = [input for input in f_graph.nodes if input.op_type == _INPUT_DUMMY_OP_NAME]

        if self.breadth_first_visitor(
            visitor,
            starting_nodes=function_inputs,
            visited_nodes=visited_nodes,
            iterate_functions=iterate_functions,
            escape_functions=False,
            max_depth=depth-1 if depth is not None else None):
          return True

      for output_tensor in n.outputs:
        for consumer in output_tensor.consumers():
          if consumer not in visited_nodes and \
              consumer not in starting_nodes_set:
            if depth is not None and depth <= 0:
              return True
            nodes_queue.put((consumer,
                             output_tensor,
                             depth-1 if depth is not None else None))

    if escape_functions and function_graph_names_set:
      function_invocation_ops = self.nodes_iterator(
          predicate=lambda f: (f.op_type in node.PARTITIONED_CALL_OP_TYPES and
            f.get_attr("f").name in function_graph_names_set),
          iterate_functions=True)
      for function_invocation_op in function_invocation_ops:
        function_output_consumers = set()
        for output_tensor in function_invocation_op.outputs:
          function_output_consumers.update(output_tensor.consumers())
        if self.breadth_first_visitor(
            visitor,
            starting_nodes=function_output_consumers,
            visited_nodes=visited_nodes,
            iterate_functions=iterate_functions,
            escape_functions=True,
            max_depth=depth-1 if depth is not None else None):
          return True
    return False

  def backwards_breadth_first_visitor(
      self,
      visitor,  # type: callable
      starting_nodes=None,  # type: Iterable[node.Node]
      visited_nodes=None,  # type: set
      iterate_functions=False,  # type: bool
      escape_functions=False,  # type: bool
      max_depth=None
      ):  # type: (...) -> None
    """
    Visit all nodes reachable from a starting set in the order of a
    backwards breadth-first traversal (going from node to input edges).
    If visitor gets to a function call, and iterate_functions is True,
    it will iterate all function nodes first and then continue with
    remaining nodes in the graph.
    Invokes a callback at each node visited.

    Args:
      visitor: Possibly-stateful callback to be invoked on each node reached
      starting_nodes: List of starting nodes.
      visited_nodes: Optional set of nodes to skip iterating over.
      iterate_functions: Indicates if we should also go inside functions if one
          is found in the graph.
      escape_functions: If iteration started in a function graph, indicates
          that we should also iterate through all function callers up
          the stack.
      max_depth: Maximum depth to iterate up to. If None, iteration will
          continue until boundary of the graph is reached.
    Returns:
      True if iteration was interrupted by visitor, otherwise False.
    """
    if not starting_nodes:
      raise ValueError("starting_nodes is not provided")

    nodes_queue = queue.Queue()
    function_graph_names_set = set()

    if visited_nodes is None:
      visited_nodes = set()

    starting_nodes_set = set()
    for n in starting_nodes:
      starting_nodes_set.add(n)
      nodes_queue.put((n, max_depth))

    while not nodes_queue.empty():
      (n, depth) = nodes_queue.get()

      if n in visited_nodes:
        continue

      if n.op_type != _INPUT_DUMMY_OP_NAME:
        if visitor(n):
          return True

      if escape_functions and isinstance(n.graph, function_graph.FunctionGraph):
        function_graph_names_set.add(n.graph.name)

      if (iterate_functions and n.op_type in node.PARTITIONED_CALL_OP_TYPES and
          n not in starting_nodes_set):
        function_name = n.get_attr("f").name
        f_graph = self.get_function_graph_by_name(function_name)

        if self.backwards_breadth_first_visitor(
            visitor,
            starting_nodes=f_graph.output_nodes,
            visited_nodes=visited_nodes,
            iterate_functions=iterate_functions,
            escape_functions=False,
            max_depth=depth-1 if depth is not None else None):
          return True

      visited_nodes.add(n)
      for input_tensor in n.inputs:
        if (input_tensor.op not in visited_nodes and
            input_tensor.op not in starting_nodes_set):
          if depth is not None and depth <= 0:
            return True
          nodes_queue.put((input_tensor.op, depth-1 if depth is not None else None))

    if escape_functions and function_graph_names_set:
      function_invocation_ops = self.nodes_iterator(
          predicate=lambda f: (f.op_type in node.PARTITIONED_CALL_OP_TYPES and
                               f.get_attr("f").name in function_graph_names_set),
          iterate_functions=True)

      caller_ops = list(function_invocation_ops)

      if len(caller_ops) > 0:
        if self.backwards_breadth_first_visitor(
            visitor,
            starting_nodes=caller_ops,
            visited_nodes=visited_nodes,
            iterate_functions=iterate_functions,
            escape_functions=True,
            max_depth=depth-1 if depth is not None else None):
          return True

    return False

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

  def _visualize_node(
      self,
      gde_node,
      format=None,
      depth=1,
      style=True,
      name_regex="",
      negative_name_regex="",
      add_digraph_func=None,
      add_digraph_node_func=None,
      add_digraph_edge_func=None,
      depth_before=1,
      depth_after=2):
    """Return GraphViz Digraph rendering of the current and adjacent nodes.

    Args:
      gde_node: a node to visualize.
      format: GraphViz display format. In addition to that it supports
        jupyter_svg, and jupyter_interactive modes.
      depth: the maximum depth of the graph to display.
      style: whether to apply default styles.
      name_regex: only diplay nodes that have name matching this regex.
      negative_name_regex: only diplay nodes that have name not matching this
        regex.
      add_digraph_func: custom override for function for adding subraphs
        to the resulting Digraph object.
      add_digraph_node_func: custom override for function for adding nodes
        (vertices) to the resulting Digraph object.
      add_digraph_edge_func: custom override for function for adding edges
        to the resulting Digraph object.
      depth_before: number of adjacent nodes to show before the current one.
      depth_after: number of adjacent nodes to show after the current one.

    Returns:
      graphviz.dot.Digraph object with visual representtion for the current
        graph.
    """

    adjacent_nodes = set()
    adjacent_nodes.add(gde_node)
    self.backwards_breadth_first_visitor(
        adjacent_nodes.add,
        max_depth=depth_before,
        starting_nodes=[gde_node])
    self.breadth_first_visitor(
        adjacent_nodes.add,
        max_depth=depth_after,
        starting_nodes=[gde_node])

    sg = subgraph.SubGraphView(adjacent_nodes)

    def custom_add_digraph_node(digraph, name, op, attributes=None):
      attributes = []
      if op == gde_node:
        attributes.append(("fillcolor", "yellow"))
      gvw.add_digraph_node(digraph, name, op, attributes)

    if not add_digraph_node_func:
      add_digraph_node_func = custom_add_digraph_node

    return sg.visualize(
        format=format,
        depth=depth,
        name=gde_node.name,
        style=style,
        name_regex=name_regex,
        negative_name_regex=negative_name_regex,
        add_digraph_func=add_digraph_func,
        add_digraph_node_func=add_digraph_node_func,
        add_digraph_edge_func=add_digraph_edge_func)


def saved_model_to_graph(saved_model_path, # type: str
                         tag = None, # type: Union[str, List[str]]
                         include_saver = True, # type: bool
                         include_signatures = True, # type: bool
                         fallback_to_default_graph = False, # type: bool
                         ):
  # type: (...) -> Graph
  """
  Load the contents of a TensorFlow SavedModel into a Graph object.

  Args:
    saved_model_path: Path to the SavedModel's directory on disk
    tag: User-specified tags attached to the MetaGraphDef that should be
      loaded from the SavedModel. If None, verify that there is only one
      MetaGraphDef in the model and load that one.
    include_saver: If True, attach black-box information about the SavedModel's
      serialized `tf.Saver` object in the returned Graph object. Otherwise the
      returned Graph will not contain any serialized variable values, though
      it will contain variable initializers.
    include_signatures: If True, attach signature information from the
      SavedModel to the returned Graph object. Otherwise the returned graph
      will have no signatures.
    fallback_to_default_graph: If True, fallback to saved_model.meta_graphs[0]
      if specified tag is not found, and saved_model.meta_graphs[0] is the
      only graph available.

  Returns: In-memory representation of the contents of the SavedModel as a
  Graph object.
  """
  if not tf.gfile.Exists(saved_model_path):
    raise ValueError(
        "SavedModel root directory {} not found".format(saved_model_path))
  if not tf.gfile.IsDirectory(saved_model_path):
    raise ValueError(
        "SavedModel root path {} is not a directory".format(saved_model_path))

  # By convention, the main protobuf for the SavedModel is in a file called
  # "saved_model.pb"
  protobuf_file = saved_model_path + "/saved_model.pb"
  saved_model = saved_model_pb2.SavedModel()
  with tf.gfile.Open(protobuf_file, "rb") as f:
    saved_model.ParseFromString(f.read())

  # Drill down to pull out the appropriate MetaGraphDef proto
  if tag is None:
    if len(saved_model.meta_graphs) != 1:
      raise ValueError("No tags specified and there are multiple "
                       "MetaGraphDefs in the SavedModel. Please specify a "
                       "tag to select a specific MetaGraphDef")
    meta_graph = saved_model.meta_graphs[0]
  else:
    tags = set(tag) if isinstance(tag, list) else set([tag])
    matching_ixs = [
        i for i in range(len(saved_model.meta_graphs))
        if tags == set(saved_model.meta_graphs[i].meta_info_def.tags)
    ]
    if len(matching_ixs) != 1:
      print(f"WARNING: exact match for tags {tags} is not found")
      matching_ixs = [
          i for i in range(len(saved_model.meta_graphs))
          if tags.issubset(set(saved_model.meta_graphs[i].meta_info_def.tags))
      ]
    if not matching_ixs:
      if fallback_to_default_graph and len(saved_model.meta_graphs) == 1:
        print(
            f"WARNING: specified tags {tags} are not found, using default one")
        meta_graph = saved_model.meta_graphs[0]
      else:
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

  object_graph_def = None
  if meta_graph.object_graph_def:
    object_graph_def=meta_graph.object_graph_def

  stripped_op_list = None
  if meta_graph.meta_info_def.stripped_op_list:
    stripped_op_list = meta_graph.meta_info_def.stripped_op_list

  return Graph(graph_def,
               name=meta_graph.meta_info_def.meta_graph_version,
               collections=collections,
               saver_info=saver_info,
               signature_info=signature_info,
               object_graph_def=object_graph_def,
               stripped_op_list=stripped_op_list,
               asset_file_def=meta_graph.asset_file_def)

################################################################################
# Stuff below this line is private to this file.


def _decode_graph(graph):
  # type: (tf.Graph) -> Dict[str, List[Tuple[tf.DType, tf.TensorShape]]]
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
  output_map = {op.name: [(t.dtype, t.shape) for t in op.outputs]
                for op in graph.get_operations()}
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
