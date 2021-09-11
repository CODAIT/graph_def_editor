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
"""Default GraphViz styles to use."""


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
