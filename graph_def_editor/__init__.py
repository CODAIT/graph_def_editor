# Copyright 2019 IBM. All Rights Reserved.
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
"""GDE: A GraphDef Editor for TensorFlow

A version of the old [`contrib.graph_editor`](https://github.com/tensorflow/tensorflow/tree/r1.12/tensorflow/contrib/graph_editor) API that operates over serialized TensorFlow graphs represented as GraphDef protocol buffer messages.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from graph_def_editor.edit import *
from graph_def_editor.graph import *
from graph_def_editor.match import *
from graph_def_editor.node import *
from graph_def_editor.reroute import *
from graph_def_editor.select import *
from graph_def_editor.subgraph import *
from graph_def_editor.transform import *
from graph_def_editor.util import *
from graph_def_editor.variable import *
# pylint: enable=wildcard-import

# Other parts go under sub-packages
from graph_def_editor import rewrite

# some useful aliases
# pylint: disable=g-bad-import-order
from graph_def_editor import subgraph as _subgraph
from graph_def_editor import util as _util
# pylint: enable=g-bad-import-order
ph = _util.make_placeholder_from_dtype_and_shape
sgv = _subgraph.make_view
sgv_scope = _subgraph.make_view_from_scope

del absolute_import
del division
del print_function
