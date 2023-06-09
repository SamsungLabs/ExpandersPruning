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

class SparseGraphConv2d(nn.Module):
    """ Sparse convolutional layer corresponding to graph_model. """

    def __init__(self, graph_model, in_channels, out_channels, degree, kernel_size, \
                 stride=1, padding=0, dilation=1, groups=1):
        super().__init__()

        def backward_hook(grad):
            cloned_grad = grad.clone()
            return cloned_grad.mul_(self.mask_on_device)

        def build_mask():
            """
            Constructs a NetworkX graph object according to graph_model.
            Returns binary mask for layer weights based on adjacency matrix of graph.
            """

            graph = graph_factory.generate_graph(graph_model, in_channels, out_channels, degree)
            biadj_matrix = nx.bipartite.biadjacency_matrix(graph, range(in_channels))
            rows, cols = biadj_matrix.nonzero()

            mask = torch.zeros(out_channels, in_channels, 1, 1)
            for row, col in zip(rows, cols):
                mask[col][row][0][0] = 1

            return mask.repeat(1, 1, kernel_size, kernel_size)

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, \
                                      dilation, groups, bias=False)

        self.register_buffer("mask_on_device", build_mask())

        """ Apply mask at initialization and backward pass. """
        self.conv2d.weight.data.mul_(self.mask_on_device)
        self.conv2d.weight.register_hook(backward_hook)

    def forward(self, layer_input):
        """ Passes input through layer. """

        return self.conv2d(layer_input)
    