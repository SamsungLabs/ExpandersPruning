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

import torch
from torch import nn
import torch.autograd
import networkx as nx

from graph_factory import graph_factory

class SparseGraphLinear(nn.Module):
    """ Sparse linear layer corresponding to graph_model. """

    def __init__(self, graph_model, in_features, out_features, degree):
        super().__init__()

        def backward_hook(grad):
            cloned_grad = grad.clone()
            return cloned_grad.mul_(self.mask_on_device)

        def build_mask():
            """
            Constructs a NetworkX graph object according to graph_model.
            Returns binary mask for layer weights based on adjacency matrix of graph.
            """

            graph = graph_factory.generate_graph(graph_model, in_features, out_features, degree)
            biadj_mat = nx.bipartite.biadjacency_matrix(graph, range(in_features))
            return torch.from_numpy(biadj_mat.toarray().transpose())

        self.linear = torch.nn.Linear(in_features, out_features)

        self.register_buffer("mask_on_device", build_mask())

        """ Apply mask at initialization and backward pass. """
        self.linear.weight.data.mul_(self.mask_on_device)
        self.linear.weight.register_hook(backward_hook)

    def forward(self, layer_input):
        """ Passes input through layer. """

        return self.linear(layer_input)
    