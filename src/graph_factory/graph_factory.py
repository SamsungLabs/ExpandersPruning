"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
James Stewart (j2.stewart@samsung.com; james1995stewart@gmail.com)
Umberto Michieli (u.michieli@samsung.com)
Mete Ozay (m.ozay@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

from graph_factory.graph_utils import generate_rreg_graph
from graph_factory.graph_utils import generate_xnet_graph
from graph_factory.graph_utils import generate_random_graph

def generate_rreg(n_0, n_1, deg):
    """ Returns a random regular graph. """

    return generate_rreg_graph(n_0, n_1, deg)

def generate_xnet(n_0, n_1, deg):
    """ Returns an XNet graph. """

    return generate_xnet_graph(n_0, n_1, deg)

def generate_random(n_0, n_1, deg):
    """ Returns a random graph. """

    return generate_random_graph(n_0, n_1, deg)

def generate_graph(graph_type, n_0, n_1, deg):
    """ Returns the relevant graph given args. """

    if graph_type == "rreg":
        return generate_rreg(n_0, n_1, deg)
    if graph_type == "xnet":
        return generate_xnet(n_0, n_1, deg)
    if graph_type == "random":
        return generate_random(n_0, n_1, deg)

    raise NotImplementedError(f"Graph type {graph_type} not implemented")
    