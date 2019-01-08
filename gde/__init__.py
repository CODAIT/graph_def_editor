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
"""GDE: A GraphDef Editor for TensorFlow

A version of the old [`contrib.graph_editor`](https://github.com/tensorflow/tensorflow/tree/r1.12/tensorflow/contrib/graph_editor) API that operates over serialized TensorFlow graphs represented as GraphDef protocol buffer messages.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from gde.edit import *
from gde.graph import *
from gde.match import *
from gde.node import *
from gde.reroute import *
from gde.select import *
from gde.subgraph import *
from gde.transform import *
from gde.util import *
from gde.variable import *
# pylint: enable=wildcard-import

# some useful aliases
# pylint: disable=g-bad-import-order
from gde import subgraph as _subgraph
from gde import util as _util
# pylint: enable=g-bad-import-order
ph = _util.make_placeholder_from_dtype_and_shape
sgv = _subgraph.make_view
sgv_scope = _subgraph.make_view_from_scope

del absolute_import
del division
del print_function
