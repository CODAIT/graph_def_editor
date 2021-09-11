# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Functionality to convert graph_def_editor graph to GraphViz visualization."""

import re
import uuid

from .graphviz_style import *
import graph_def_editor.visualization.jupyter_helper as jupyter_helper


FORMAT_JUPYTER_SVG = 'jupyter_svg'
FORMAT_JUPYTER_INTERACTIVE = 'jupyter_interactive'

_CLUSTER_INDEX = 0  # index of subgraph
_ADD_DIGRAPH_FUNC = None
_ADD_DIGRAPH_NODE_FUNC = None
_ADD_DIGRAPH_EDGE_FUNC = None


graph_pref = {
    'fontcolor': '#414141',
    'style': 'rounded',
}

name_scope_graph_pref = {
    'bgcolor': '#eeeeee',
    'color': '#aaaaaa',
    'penwidth': '2',
}

non_name_scope_graph_pref = {
    'fillcolor': 'white',
    'color': 'white',
}

node_pref = {
    'style': 'filled',
    'fillcolor': 'white',
    'color': '#aaaaaa',
    'penwidth': '2',
    'fontcolor': '#414141',
}

edge_pref = {
    'color': '#aaaaaa',
    'arrowsize': '1.2',
    'penwidth': '2.5',
    'fontcolor': '#414141',
}


def add_digraph(name=None, name_scope=None, style=True):
  """Return graphviz.dot.Digraph with TensorBoard-like style."""
  try:
    import graphviz as gv
  except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
      "You need to install graphviz to be able to use this functionality. "
      "See https://graphviz.readthedocs.io/en/stable/manual.html for details.")

  digraph = gv.Digraph(name=name)
  if name_scope:
    digraph.graph_attr['label'] = name_scope
    digraph.graph_attr['tooltip'] = name_scope

  if style is False:
    return digraph

  if name_scope:
    digraph.graph_attr.update(name_scope_graph_pref)
  else:
    digraph.graph_attr.update(non_name_scope_graph_pref)
  digraph.graph_attr.update(graph_pref)
  digraph.node_attr.update(node_pref)
  digraph.edge_attr.update(edge_pref)
  return digraph


def add_digraph_node(digraph, name, op, attributes=None):
  """Adds a node to digraph."""
  label = name.split('/')[-1]
  tooltip = name
  # For possible attribute values see:
  # https://graphviz.org/doc/info/attrs.html
  if attributes is None:
    attributes = []
  if op is not None:
    tooltip += ':' + op.op_type
    if 'PartitionedCall' in op.op_type:
      try:
        label = '{}\n{}:{}'.format(label, 'f', op.get_attr('f').name)
      except ValueError:
        pass
      # For example:
      # attributes.append(('fillcolor', 'green'))
  digraph.node(name, label=label, tooltip=tooltip, _attributes=attributes)


def add_digraph_edge(digraph, from_node, to_node, label=None, attributes=None):
  """Adds an edge to digraph."""
  if attributes is None:
    attributes = []
  digraph.edge(from_node, to_node, label=label, _attributes=attributes)


def nested_dict(dict_, keys, val):
  """Assign value to dictionary."""
  cloned = dict_.copy()
  if len(keys) == 1:
    cloned[keys[0]] = val
    return cloned
  dd = cloned[keys[0]]
  for k in keys[1:len(keys) - 1]:
    dd = dd[k]
  last_key = keys[len(keys) - 1]
  dd[last_key] = val
  return cloned


def node_abs_paths(node):
  """Return absolute node path name."""
  node_names = node.name.split('/')
  return ['/'.join(node_names[0:i + 1]) for i in range(len(node_names))]


def node_table(gde_graph, depth=1, match_func=None):
  """Return dictionary of node."""
  table = {}
  ops_table = {}
  max_depth = depth
  ops = gde_graph.nodes
  for depth_i in range(max_depth):
    for op in ops:
      abs_paths = node_abs_paths(op)
      if depth_i >= len(abs_paths):
        continue
      if match_func and not match_func(op.name):
        continue
      ops_table[op.name] = op
      ps = abs_paths[:depth_i + 1]
      if len(ps) == 1:
        key = '/'.join(abs_paths[0:depth_i + 1])
        if not key in table:
          table[key] = {}
      else:
        table = nested_dict(table, ps, {})
  return table, ops_table


def tensor_shape(gde_tensor, depth=1):
  """Return node and the children."""
  outpt_name = gde_tensor.name
  if len(outpt_name.split('/')) < depth:
    return None
  on = '/'.join(outpt_name.split('/')[:depth])  # output node
  result = re.match(r'(.*):\d*$', on)
  if not result:
    return None
  on = result.groups()[0]
  if gde_tensor.shape.ndims is None:
    return on, []
  else:
    return on, gde_tensor.shape.as_list()


def node_input_table(gde_graph, depth=1, match_func=None):
  """Return table of operations."""
  table = {}
  inpt_op_table = {}
  inpt_op_shape_table = {}
  for op in gde_graph.nodes:
    if match_func and not match_func(op.name):
      continue
    op_name = op.name.split('/')[0:depth]
    opn = '/'.join(op_name)
    if not opn in inpt_op_table:
      inpt_op_table[opn] = []
    inpt_op_list = ['/'.join(input_tensor.op.name.split('/')[0:depth]) \
        for input_tensor in op.inputs if not match_func or match_func(input_tensor.op.name)]
    inpt_op_table[opn].append(inpt_op_list)
    for output in op.outputs:
      for i in range(depth):
        shape = tensor_shape(output, depth=i + 1)
        if shape:
          inpt_op_shape_table[shape[0]] = shape[1]
  for opn in inpt_op_table.keys():
    t_l = []
    for ll in inpt_op_table[opn]:
      list.extend(t_l, ll)
    table[opn] = list(set(t_l))
  return table, inpt_op_shape_table


def add_nodes(node_table, ops_table, name=None, name_scope=None, style=True):
  """Add TensorFlow graph's nodes to graphviz.dot.Digraph."""
  global _CLUSTER_INDEX
  global _ADD_DIGRAPH_FUNC
  global _ADD_DIGRAPH_NODE_FUNC
  if name:
    digraph = _ADD_DIGRAPH_FUNC(name=name, name_scope=name_scope, style=style)
  else:
    digraph = _ADD_DIGRAPH_FUNC(
        name=str(uuid.uuid4().hex.upper()[0:6]),
        name_scope=name_scope,
        style=style)
  graphs = []
  for key, value in node_table.items():
    if len(value) > 0:
      sg = add_nodes(
          value,
          ops_table,
          name='cluster_%i' % _CLUSTER_INDEX,
          name_scope=key.split('/')[-1],
          style=style)
      op = ops_table.get(key, None)
      _ADD_DIGRAPH_NODE_FUNC(sg, key, op)
      _CLUSTER_INDEX += 1
      graphs.append(sg)
    else:
      op = ops_table.get(key, None)
      label = key.split('/')[-1]
      _ADD_DIGRAPH_NODE_FUNC(digraph, key, op)

  for tg in graphs:
    digraph.subgraph(tg)
  return digraph


def edge_label(shape):
  """Returns texts of graph's edges."""
  if len(shape) == 0:
    return ''
  if shape[0] is None:
    label = '?'
  else:
    label = '%i' % shape[0]
  for s in shape[1:]:
    if s is None:
      label += '×?'
    else:
      label += u'×%i' % s
  return label


def add_edges(digraph, node_inpt_table, node_inpt_shape_table):
  """Add graph's edges to graphviz.dot.Digraph."""
  global _ADD_DIGRAPH_EDGE_FUNC
  for node, node_inputs in node_inpt_table.items():
    if re.match(r'\^', node):
      continue
    for ni in node_inputs:
      if ni == node:
        continue
      if re.match(r'\^', ni):
        continue
      if not ni in node_inpt_shape_table:
        _ADD_DIGRAPH_EDGE_FUNC(digraph, ni, node)
      else:
        shape = node_inpt_shape_table[ni]
        _ADD_DIGRAPH_EDGE_FUNC(digraph, ni, node, label=edge_label(shape))
  return digraph


def match_func(name_regex, negative_name_regex):
  name_re = None
  if name_regex:
    name_re = re.compile(name_regex)

  negative_name_re = None
  if negative_name_regex:
    negative_name_re = re.compile(negative_name_regex)

  def _matches(node_name):
    return bool(
        (not name_re or name_re.search(node_name)) and
        (not negative_name_re or not negative_name_re.search(node_name)))

  return _matches


def board(gde_graph,
          depth=1,
          name='G',
          style=True,
          name_regex='',
          negative_name_regex='',
          add_digraph_func=None,
          add_digraph_node_func=None,
          add_digraph_edge_func=None):
  """Return GraphViz Digraph rendering of the specified graph.

  Args:
    depth: the maximum depth of the graph to display.
    name: graph name.
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

  Returns:
    graphviz.dot.Digraph object with visual representtion for the specified
      graph.
  """
  global _ADD_DIGRAPH_FUNC
  global _ADD_DIGRAPH_NODE_FUNC
  global _ADD_DIGRAPH_EDGE_FUNC
  global _CLUSTER_INDEX
  _CLUSTER_INDEX = 0
  _ADD_DIGRAPH_FUNC = add_digraph_func if add_digraph_func is not None else add_digraph
  _ADD_DIGRAPH_NODE_FUNC = add_digraph_node_func if add_digraph_node_func is not None else add_digraph_node
  _ADD_DIGRAPH_EDGE_FUNC = add_digraph_edge_func if add_digraph_edge_func is not None else add_digraph_edge
  _node_name_matches_func = match_func(
      name_regex, negative_name_regex)

  _node_table, _ops_table = node_table(
      gde_graph, depth=depth, match_func=_node_name_matches_func)
  _node_inpt_table, _node_inpt_shape_table = node_input_table(
      gde_graph, depth=depth, match_func=_node_name_matches_func)
  digraph = add_nodes(_node_table, _ops_table, name=name, style=style)
  digraph = add_edges(digraph, _node_inpt_table, _node_inpt_shape_table)
  return digraph


def visualize(
    gde_graph,
    format=None,
    depth=1,
    name='G',
    style=True,
    name_regex='',
    negative_name_regex='',
    add_digraph_func=None,
    add_digraph_node_func=None,
    add_digraph_edge_func=None):
  """Return GraphViz Digraph rendering of the specified graph.

  Args:
    gde_graph: Graph to display.
    format: GraphViz display format (see https://graphviz.org/docs/outputs/).
      In addition to that it supports jupyter_svg, and jupyter_interactive modes
    depth: the maximum depth of the graph to display.
    name: graph name.
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

  Returns:
    graphviz.dot.Digraph object with visual representtion for the specified
      graph.
  """
  dg = board(
      gde_graph,
      depth=depth,
      name=name,
      style=style,
      name_regex=name_regex,
      negative_name_regex=negative_name_regex,
      add_digraph_func=add_digraph_func,
      add_digraph_node_func=add_digraph_node_func,
      add_digraph_edge_func=add_digraph_edge_func)

  if format is None:
    return dg
  elif format == FORMAT_JUPYTER_SVG:
    return jupyter_helper.jupyter_show_as_svg(dg)
  elif format == FORMAT_JUPYTER_INTERACTIVE:
    return jupyter_helper.jupyter_pan_and_zoom(dg)
  else:
    return dg.pipe(format=format)
