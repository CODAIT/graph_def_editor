# Copyright 2018 IBM. All Rights Reserved.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for gde
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import sys
if sys.version >= '3':
  from typing import Any, List

import numpy as np
from six import iteritems, string_types
import tensorflow as tf

from graph_def_editor import graph, node, tensor


__all__ = [
  "make_list_of_op",
  "make_list_of_t",
  "get_generating_ops",
  "get_consuming_ops",
  "ControlOutputs",
  "placeholder_name",
  "make_placeholder_from_tensor",
  "make_placeholder_from_dtype_and_shape",
  "load_variables_to_tf_graph",
  "make_const",
  "make_placeholder",
  "make_simple_binary_op"
]


# The graph editor sometimes need to create placeholders, they are named
# "geph_*". "geph" stands for Graph-Editor PlaceHolder.
_DEFAULT_PLACEHOLDER_PREFIX = "geph"


def concatenate_unique(la, lb):
  """Add all the elements of `lb` to `la` if they are not there already.

  The elements added to `la` maintain ordering with respect to `lb`.

  Args:
    la: List of Python objects.
    lb: List of Python objects.
  Returns:
    `la`: The list `la` with missing elements from `lb`.
  """
  la_set = set(la)
  for l in lb:
    if l not in la_set:
      la.append(l)
      la_set.add(l)
  return la


# TODO(fkp): very generic code, it should be moved in a more generic place.
class ListView(object):
  """Immutable list wrapper.

  This class is strongly inspired by the one in tf.Operation.
  """

  def __init__(self, list_):
    if not isinstance(list_, list):
      raise TypeError("Expected a list, got: {}.".format(type(list_)))
    self._list = list_

  def __iter__(self):
    return iter(self._list)

  def __len__(self):
    return len(self._list)

  def __bool__(self):
    return bool(self._list)

  # Python 3 wants __bool__, Python 2.7 wants __nonzero__
  __nonzero__ = __bool__

  def __getitem__(self, i):
    return self._list[i]

  def __add__(self, other):
    if not isinstance(other, list):
      other = list(other)
    return list(self) + other

  def __str__(self):
    return "ListView[{}]".format(self._list)


# TODO(fkp): very generic code, it should be moved in a more generic place.
def is_iterable(obj):
  """Return true if the object is iterable."""
  if isinstance(obj, node.Node):
    return False
  try:
    _ = iter(obj)
  except Exception:  # pylint: disable=broad-except
    return False
  return True


def flatten_tree(tree, leaves=None):
  """Flatten a tree into a list.
  Args:
    tree: iterable or not. If iterable, its elements (child) can also be
      iterable or not.
    leaves: list to which the tree leaves are appended (None by default).
  Returns:
    A list of all the leaves in the tree.
  """
  if leaves is None:
    leaves = []
  if isinstance(tree, dict):
    for _, child in iteritems(tree):
      flatten_tree(child, leaves)
  elif is_iterable(tree):
    for child in tree:
      flatten_tree(child, leaves)
  else:
    leaves.append(tree)
  return leaves


def transform_tree(tree, fn, iterable_type=tuple):
  """Transform all the nodes of a tree.
  Args:
    tree: iterable or not. If iterable, its elements (child) can also be
      iterable or not.
    fn: function to apply to each leaves.
    iterable_type: type use to construct the resulting tree for unknown
      iterable, typically `list` or `tuple`.
  Returns:
    A tree whose leaves has been transformed by `fn`.
    The hierarchy of the output tree mimics the one of the input tree.
  """
  if is_iterable(tree):
    if isinstance(tree, dict):
      res = tree.__new__(type(tree))
      res.__init__(
        (k, transform_tree(child, fn)) for k, child in iteritems(tree))
      return res
    elif isinstance(tree, tuple):
      # NamedTuple?
      if hasattr(tree, "_asdict"):
        res = tree.__new__(type(tree), **transform_tree(tree._asdict(), fn))
      else:
        res = tree.__new__(type(tree),
                           (transform_tree(child, fn) for child in tree))
      return res
    elif isinstance(tree, collections.Sequence):
      res = tree.__new__(type(tree))
      res.__init__(transform_tree(child, fn) for child in tree)
      return res
    else:
      return iterable_type(transform_tree(child, fn) for child in tree)
  else:
    return fn(tree)


def check_graphs(*args):
  """Check that all the elements in args belong to the same graph.

  Args:
    *args: a list of object with a obj.graph property.
  Raises:
    ValueError: if all the elements do not belong to the same graph.
  """
  g = None
  for i, sgv in enumerate(args):
    if g is None and sgv.graph is not None:
      g = sgv.graph
    elif sgv.graph is not None and sgv.graph is not g:
      raise ValueError("Argument[{}]: Wrong graph!".format(i))


def get_unique_graph(tops, check_types=None, none_if_empty=False):
  """Return the unique graph used by the all the elements in tops.

  Args:
    tops: list of elements to check (usually a list of `gde.Tensor`). Or a
      `gde.Graph`.
    check_types: check that the element in tops are of given type(s). If None,
      the types (`gde.Node`, `gde.Tensor`) are used.
    none_if_empty: don't raise an error if tops is an empty list, just return
      None.
  Returns:
    The unique graph used by all the tops.
  Raises:
    TypeError: if tops is not a iterable of `gde.Node`.
    ValueError: if the graph is not unique.
  """
  if isinstance(tops, graph.Graph):
    return tops
  if not is_iterable(tops):
    raise TypeError("{} is not iterable".format(type(tops)))
  if check_types is None:
    check_types = (node.Node, tensor.Tensor,)
  elif not is_iterable(check_types):
    check_types = (check_types,)
  g = None
  for op in tops:
    if not isinstance(op, check_types):
      raise TypeError("Expected a type in ({}), got: {}".format(", ".join([str(
          t) for t in check_types]), type(op)))
    if g is None:
      g = op.graph
    elif g is not op.graph:
      raise ValueError("Operation {} does not belong to given graph".format(op))
  if g is None and not none_if_empty:
    raise ValueError("Can't find the unique graph of an empty list")
  return g


def make_list_of_op(ops, check_graph=True, allow_graph=True, ignore_ts=False):
  """Convert ops to a list of `gde.Node`.

  Args:
    ops: can be an iterable of `gde.Node`, a `gde.Graph` or a single
      Node.
    check_graph: if `True` check if all the nodes belong to the same graph.
    allow_graph: if `False` a `gde.Graph` cannot be converted.
    ignore_ts: if True, silently ignore `gde.Tensor`.
  Returns:
    A newly created list of `gde.Node`.
  Raises:
    TypeError: if ops cannot be converted to a list of `gde.Node` or,
     if `check_graph` is `True`, if all the ops do not belong to the
     same graph.
  """
  if isinstance(ops, graph.Graph):
    if allow_graph:
      return ops.nodes
    else:
      raise TypeError("allow_graph is False: cannot convert a gde.Graph.")
  else:
    if not is_iterable(ops):
      ops = [ops]
    if not ops:
      return []
    if check_graph:
      check_types = None if ignore_ts else node.Node
      get_unique_graph(ops, check_types=check_types)
    return [op for op in ops if isinstance(op, node.Node)]


def make_list_of_t(ts, check_graph=True, allow_graph=True, ignore_ops=False):
  # type: (...) -> List[tensor.Tensor]
  """Convert ts to a list of `gde.Tensor`.

  Args:
    ts: can be an iterable of `gde.Tensor`, a `gde.Graph` or a single tensor.
    check_graph: if `True` check if all the tensors belong to the same graph.
    allow_graph: if `False` a `gde.Graph` cannot be converted.
    ignore_ops: if `True`, silently ignore `gde.Node`.
  Returns:
    A newly created list of `gde.Tensor`.
  Raises:
    TypeError: if `ts` cannot be converted to a list of `gde.Tensor` or,
     if `check_graph` is `True`, if all the ops do not belong to the same graph.
  """
  if isinstance(ts, graph.Graph):
    if allow_graph:
      return ts.tensors
    else:
      raise TypeError("allow_graph is False: cannot convert a gde.Graph.")
  else:
    if not is_iterable(ts):
      ts = [ts]
    if not ts:
      return []
    if check_graph:
      check_types = None if ignore_ops else (tensor.Tensor,)
      get_unique_graph(ts, check_types=check_types)
    return [t for t in ts if isinstance(t, tensor.Tensor)]


def get_generating_ops(ts):
  """Return all the generating ops of the tensors in `ts`.

  Args:
    ts: a list of `gde.Tensor`
  Returns:
    A list of all the `gde.Node` objects that represent the generating
    `gde.Node`s of the tensors in `ts`.
  Raises:
    TypeError: if `ts` cannot be converted to a list of `gde.Tensor`.
  """
  ts = make_list_of_t(ts, allow_graph=False)
  return [t.node for t in ts]


def get_consuming_ops(ts):
  """Return all the consuming ops of the tensors in ts.

  Args:
    ts: a list of `gde.Tensor`
  Returns:
    A list of all the `gde.Node` objects that represent the consuming
    `gde.Node`s of the tensors in `ts`.
  Raises:
    TypeError: if ts cannot be converted to a list of `gde.Tensor`.
  """
  ts = make_list_of_t(ts, allow_graph=False)
  ops = []
  for t in ts:
    for op in t.consumers():
      if op not in ops:
        ops.append(op)
  return ops


class ControlOutputs(object):
  """The control outputs topology."""

  def __init__(self,
               g # type: graph.Graph
               ):
    """Create a dictionary of control-output dependencies.

    Args:
      g: a `gde.Graph`.
    Returns:
      A dictionary where a key is `gde.Node` object and the corresponding value
      is a list of all the ops which have the keys one of their control-input
      dependencies.
    Raises:
      TypeError: graph is not a `gde.Graph`.
    """
    if not isinstance(g, graph.Graph):
      raise TypeError("Expected a gde.Graph, got: {}".format(type(g)))
    self._control_outputs = {}
    self._graph = g
    self._version = None
    self._build()

  def update(self):
    """Update the control outputs if the graph has changed."""
    if self._version != self._graph.version:
      self._build()
    return self

  def _build(self):
    """Build the control outputs dictionary."""
    self._control_outputs.clear()
    for n in self._graph.nodes:
      for control_input in n.control_inputs:
        if control_input not in self._control_outputs:
          self._control_outputs[control_input] = []
        if n not in self._control_outputs[control_input]:
          self._control_outputs[control_input].append(n)
    self._version = self._graph.version

  def get_all(self):
    return self._control_outputs

  def get(self, op):
    """return the control outputs of op."""
    if op in self._control_outputs:
      return self._control_outputs[op]
    else:
      return ()

  @property
  def graph(self):
    return self._graph


def scope_finalize(scope):
  if scope and scope[-1] != "/":
    scope += "/"
  return scope


def scope_dirname(scope):
  slash = scope.rfind("/")
  if slash == -1:
    return ""
  return scope[:slash + 1]


def scope_basename(scope):
  slash = scope.rfind("/")
  if slash == -1:
    return scope
  return scope[slash + 1:]


def placeholder_name(t=None, scope=None, prefix=_DEFAULT_PLACEHOLDER_PREFIX):
  """Create placeholder name for the graph editor.

  Args:
    t: optional `gde.Tensor` on which the placeholder operation's name will be
      based on
    scope: absolute scope with which to prefix the placeholder's name. None
      means that the scope of t is preserved. "" means the root scope.
    prefix: placeholder name prefix.
  Returns:
    A new placeholder name prefixed by "geph". Note that "geph" stands for
      Graph Editor PlaceHolder. This convention allows to quickly identify the
      placeholder generated by the Graph Editor.
  Raises:
    TypeError: if t is not None or a `gde.Tensor`.
  """
  if scope is not None:
    scope = scope_finalize(scope)
  if t is not None:
    if not isinstance(t, tensor.Tensor):
      raise TypeError("Expected a gde.Tensor, got: {}".format(type(t)))
    op_dirname = scope_dirname(t.node.name)
    op_basename = scope_basename(t.node.name)
    if scope is None:
      scope = op_dirname

    if op_basename.startswith("{}__".format(prefix)):
      ph_name = op_basename
    else:
      ph_name = "{}__{}_{}".format(prefix, op_basename, t.value_index)

    return scope + ph_name
  else:
    if scope is None:
      scope = ""
    return "{}{}".format(scope, prefix)


def make_placeholder_from_tensor(
        g, # type: graph.Graph
        t, # type: tensor.Tensor
        scope=None,
        prefix=_DEFAULT_PLACEHOLDER_PREFIX
        ):
  """Create a `gde.Node` representing a `tf.placeholder` for the Graph Editor.

  Note that the correct graph scope must be set by the calling function.

  Args:
    g: A `gde.Graph` object in which the placeholder should go.
    t: a `gde.Tensor` whose name will be used to create the placeholder
    scope: absolute scope within which to create the placeholder. None
      means that the scope of `t` is preserved. `""` means the root scope.
    prefix: placeholder name prefix.
  Returns:
    A newly created `gde.Node` that represents the `tf.placeholder`.
  Raises:
    TypeError: if `t` is not `None` or a `gde.Tensor`.
  """
  return make_placeholder(g, dtype=t.dtype, shape=t.shape,
                          name=placeholder_name(t, scope=scope,
                                                prefix=prefix))


def make_placeholder_from_dtype_and_shape(g, dtype, shape=None, scope=None,
                                          prefix=_DEFAULT_PLACEHOLDER_PREFIX):
  """Create a `gde.Node` representing a `tf.placeholder` for the Graph Editor.

  Note that the correct graph scope must be set by the calling function.
  The placeholder is named using the function placeholder_name (with no
  tensor argument).

  Args:
    g: A `gde.Graph` object in which the placeholder should go.
    dtype: the tensor type.
    shape: the tensor shape (optional).
    scope: absolute scope within which to create the placeholder. None
      means that the scope of t is preserved. "" means the root scope.
    prefix: placeholder name prefix.
  Returns:
    A newly created `gde.Node`.
  """
  return make_placeholder(g,
                          dtype=dtype, shape=shape,
                          name=placeholder_name(scope=scope, prefix=prefix))


_INTERNAL_VARIABLE_RE = re.compile(r"^__\w+__$")


def get_predefined_collection_names():
  """Return all the predefined collection names."""
  return [getattr(tf.GraphKeys, key) for key in dir(tf.GraphKeys)
          if not _INTERNAL_VARIABLE_RE.match(key)]


def find_corresponding_elem(target, dst_graph, dst_scope="", src_scope=""):
  """Find corresponding op/tensor in a different graph.

  Args:
    target: A `gde.Tensor` or a `gde.Node` belonging to the original graph.
    dst_graph: The graph in which the corresponding graph element must be found.
    dst_scope: A scope which is prepended to the name to look for.
    src_scope: A scope which is removed from the original of `target` name.

  Returns:
    The corresponding `gde.Tensor` or a `gde.Node`.

  Raises:
    ValueError: if `src_name` does not start with `src_scope`.
    TypeError: if `target` is not a `gde.Tensor` or a `gde.Node`
    KeyError: If the corresponding graph element cannot be found.
  """
  src_name = target.name
  if src_scope:
    src_scope = scope_finalize(src_scope)
    if not src_name.startswidth(src_scope):
      raise ValueError("{} does not start with {}".format(src_name, src_scope))
    src_name = src_name[len(src_scope):]

  dst_name = src_name
  if dst_scope:
    dst_scope = scope_finalize(dst_scope)
    dst_name = dst_scope + dst_name

  if isinstance(target, tensor.Tensor):
    return dst_graph.get_tensor_by_name(dst_name)
  if isinstance(target, node.Node):
    return dst_graph[dst_name]
  raise TypeError("Expected gde.Tensor or gde.Node, got: {}", type(target))


def find_corresponding(targets, dst_graph, dst_scope="", src_scope=""):
  """Find corresponding ops/tensors in a different graph.

  `targets` is a Python tree, that is, a nested structure of iterable
  (list, tuple, dictionary) whose leaves are instances of
  `gde.Tensor` or `gde.Node`

  Args:
    targets: A Python tree containing `gde.Tensor` or `gde.Node`
      belonging to the original graph.
    dst_graph: The graph in which the corresponding graph element must be found.
    dst_scope: A scope which is prepended to the name to look for.
    src_scope: A scope which is removed from the original of `top` name.

  Returns:
    A Python tree containing the corresponding `gde.Tensor` or a `gde.Node`.

  Raises:
    ValueError: if `src_name` does not start with `src_scope`.
    TypeError: if `top` is not a `gde.Tensor` or a `gde.Node`
    KeyError: If the corresponding graph element cannot be found.
  """
  def func(top):
    return find_corresponding_elem(top, dst_graph, dst_scope, src_scope)
  return transform_tree(targets, func)


def _python_type_to_attr_list_elem(
        list_value, # type: tf.AttrValue.ListValue
        elem # type: Any
  ):
  """
  Subroutine of python_type_to_attr_value(). Converts one element of a Python
  list to one element of a `tf.AttrValue.ListValue` protobuf.

  Args:
    list_value: ListValue proto being populated for use within an AttrValue
      proto. Modified in place.
    elem: Original value to convert.
  """
  if isinstance(elem, string_types):
    list_value.s.append(tf.compat.as_bytes(elem))
  # Must check for bool before int because bool is a subclass of int in Python
  elif isinstance(elem, bool):
    list_value.b.append(elem)
  elif isinstance(elem, int):
    list_value.i.append(elem)
  elif isinstance(elem, float):
    list_value.f.append(elem)
  elif isinstance(elem, tf.DType):
    list_value.type.append(elem.as_datatype_enum)
  elif isinstance(elem, tf.TensorShape):
    list_value.shape.add().CopyFrom(elem.as_proto())
  elif isinstance(elem, np.ndarray):
    list_value.tensor.add().CopyFrom(tf.make_tensor_proto(values=elem))
  # TODO(frreiss): Populate the "func" field of the union here
  else:
    raise ValueError("Don't know how to convert a {} to "
                     "tf.AttrValue.ListValue".format(type(elem)))


def python_type_to_attr_value(value #type: Any
                              ):
  # type (...) -> tf.AttrValue
  """
  Convert a Python object or scalar value to a TensorFlow `tf.AttrValue`
  protocol buffer message.

  Args:
    value: Python object to be converted

  Returns:
    An AttrValue object that wraps the contents of `value` in the most
    appropriate way available.
  """
  if isinstance(value, list) or isinstance(value, tuple):
    if 0 == len(value):
      return tf.AttrValue(list=tf.AttrValue.ListValue())
    else:
      # Nonempty list
      list_value = tf.AttrValue.ListValue()
      for elem in value:
        # TODO(frreiss): Should we disallow heterogeneous types in lists?
        _python_type_to_attr_list_elem(list_value, elem)
      return tf.AttrValue(list=list_value)
  elif isinstance(value, tf.AttrValue):
    # TODO(frreiss): Should this case result in an error?
    return value
  # Scalar types, in the order they appear in the .proto file
  elif isinstance(value, string_types):
    return tf.AttrValue(s=tf.compat.as_bytes(value))
  # Must check for bool before int because bool is a subclass of int in Python
  elif isinstance(value, bool):
    return tf.AttrValue(b=value)
  elif isinstance(value, int):
    return tf.AttrValue(i=value)
  elif isinstance(value, float):
    return tf.AttrValue(f=value)
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


def attr_value_to_python_type(attr_value # type: tf.AttrValue
                              ):
  # type (...) -> Any
  """
  Inverse of python_type_to_attr_value().

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


def load_variables_to_tf_graph(g # type: graph.Graph
                               ):
  """
  Convenience function to load all variables present in a `gde.Graph` into
  the current default TensorFlow graph, without generating a MetaGraphDef.
  Also adds those variables to the appropriate TensorFlow collections.

  Args:
    g: `gde.Graph` object from which all variables and variable collections
      should be loaded
  """
  for var_name in g.variable_names:
    var = g.get_variable_by_name(var_name)
    tf_var = tf.Variable.from_proto(var.to_proto())
    tf.add_to_collections(var.collection_names, tf_var)


def make_const(g, # type: graph.Graph
               name, # type: str
               value, # type: np.ndarray
               uniquify_name=False # type: bool
               ):
  """
  Convenience method to add a `Const` op to a `gde.Graph`.

  Args:
    g: The graph that the node should be added to
    name: Name for the new `Const` node
    value: Value to use for the constant
    uniquify_name: if True, generate unique names by appending a numeric
      suffix in the event of a name collision. Otherwise name collisions
      result in an error.

  Returns `gde.Node` object representing the new node.
  """
  dtype = tf.as_dtype(value.dtype)
  ret = g.add_node(name, "Const", uniquify_name=uniquify_name)
  ret.add_attr("dtype", dtype)
  ret.add_attr("value", value)
  ret.set_outputs_from_pairs([(dtype, tf.TensorShape(value.shape))])
  return ret


def make_placeholder(g, # type: graph.Graph
                     name, # type: str
                     dtype, # type: tf.DType
                     shape, #type: tf.TensorShape
                     uniquify_name=False # type: bool
                     ):
  """
  Convenience method to add a `Placeholder` op to a `gde.Graph`.

  Args:
    g: The graph that the new node should be added to
    name: Name for the new node
    dtype: `tf.DType` holding the dtype of the placeholder
    shape: `tf.TensorShape` representing the shape returned by the placeholder
    uniquify_name: if True, generate unique names by appending a numeric
      suffix in the event of a name collision. Otherwise name collisions
      result in an error.

  Returns `gde.Node` object representing the new node.
  """
  ret = g.add_node(name, "Placeholder", uniquify_name=uniquify_name)
  ret.add_attr("dtype", dtype)
  ret.set_outputs_from_pairs([(dtype, shape)])
  return ret


def make_identity(g, # type: graph.Graph
                  name, # type: str
                  input, # type: tensor.Tensor
                  uniquify_name=False # type: bool
                  ):
  """
  Convenience method to add an `Identity` op to a `gde.Graph`.

  Args:
    g: The graph that the new node should be added to
    name: Name for the new node
    input: `tensor.Tensor` Input Tensor to be returned by the identity op
    uniquify_name: if True, generate unique names by appending a numeric
      suffix in the event of a name collision. Otherwise name collisions
      result in an error.

  Returns `gde.Node` object representing the new node.
  """
  ret = g.add_node(name, "Identity", uniquify_name=uniquify_name)
  ret.set_inputs([input])
  ret.set_outputs_from_pairs([(input.dtype, input.shape)])
  ret.add_attr("T", input.dtype)
  return ret


def make_simple_binary_op(g, # type: graph.Graph
                          name, # type: str
                          op_name, # type: str
                          input_1, # type: tensor.Tensor
                          input_2, # type: tensor.Tensor
                          dtype=None, # type: tf.DType
                          uniquify_name=False # type: bool
                          ):
  """
  Convenience method to cover the common case of binary ops. To be used with
  this pattern, ops must satisfy the following:
    * Two inputs
    * One output
    * DType of output stored in an attribute called "T"

  Args:
    g: The graph that the node should be added to
    name: Name for the new node
    op_name: Name of the op to use at this node
    input_1: First input tensor
    input_2: Second input tensor
    dtype: dtype returned; if None, will use the dtype of input_1
    uniquify_name: if True, generate unique names by appending a numeric
      suffix in the event of a name collision. Otherwise name collisions
      result in an error.

  Returns `gde.Node` object representing the new node.
  """
  if dtype is None:
    dtype = input_1.dtype
  ret = g.add_node(name, op_name, uniquify_name=uniquify_name)
  ret.add_attr("T", dtype)
  ret.set_inputs([input_1, input_2])
  ret.infer_outputs()
  return ret
