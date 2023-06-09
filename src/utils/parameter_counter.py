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
from prettytable import PrettyTable

from model_factory.layers.convolutional_layers.sparse_graph_conv2d import SparseGraphConv2d
from model_factory.layers.linear_layers.sparse_graph_linear import SparseGraphLinear

from utils.config import models

def get_layers(model):
    """ Returns model layers. """

    layers = list(model.children())

    flattened_layers = []

    if not layers:
        return model

    for layer in layers:
        try:
            if isinstance(layer, SparseGraphConv2d) or isinstance(layer, SparseGraphLinear):
                flattened_layers.extend(layer)
            else:
                flattened_layers.extend(get_layers(layer))
        except TypeError:
            if isinstance(layer, SparseGraphConv2d) or isinstance(layer, SparseGraphLinear):
                flattened_layers.append(layer)
            else:
                flattened_layers.append(get_layers(layer))

    return flattened_layers

def count_params(model, print_summary=False):
    """ Returns number of unmasked model parameters. """

    layers = get_layers(model)

    table = PrettyTable(["Layers", "Max. params", "Masked params", "Layer params"])
    total_params = 0

    for layer in layers:
        layer_masked = 0

        for name, buffer in layer.named_buffers():
            if name == "mask_on_device":
                layer_masked = (torch.numel(buffer)-torch.count_nonzero(buffer)).item()

        layer_params = 0
        layer_name = str(layer).replace(" ", "").replace("\r", "").replace("\n", "")

        for _, parameter in layer.named_parameters():
            if not parameter.requires_grad:
                continue

            layer_params += parameter.numel()

        total_params += layer_params-layer_masked

        if layer_params > 0:
            table.add_row([layer_name, layer_params, layer_masked, layer_params-layer_masked])

    if print_summary:
        print(table)

    return total_params

def count_parameters(model_type, model, print_summary=False):
    """ Returns number of unmasked model parameters if model supported. """

    if model_type in models:
        return count_params(model, print_summary)

    raise NotImplementedError(f"Could not print num parameters as model_type \
                              {model_type} is not supported.")
