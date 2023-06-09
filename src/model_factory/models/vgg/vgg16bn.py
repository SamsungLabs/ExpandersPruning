"""
This code is adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
Copyright (c) Soumith Chintala 2016.

"""

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

from typing import cast, List, Union

import torch
import torch.nn as nn

from model_factory.layers.convolutional_layers.sparse_graph_conv2d import SparseGraphConv2d
from model_factory.layers.linear_layers.sparse_graph_linear import SparseGraphLinear

class VGG(nn.Module):
    def __init__(self, graph_model, degree: int, features: nn.Module, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512) if graph_model == "fc" else SparseGraphLinear(graph_model, 512, 512, degree=degree),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(graph_model, degree: int, cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) if graph_model == "fc" \
                        else SparseGraphConv2d(graph_model, in_channels, v, degree=degree, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16_bn(graph_model, degree: int, num_classes: int = 1000) -> VGG:
    return VGG(graph_model,
               degree=degree,
               features=make_layers(graph_model, degree, [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], batch_norm=True),
               num_classes=num_classes)

def generate_cifar10(graph_model, degree):
    return vgg16_bn(graph_model, degree=degree, num_classes=10)

def generate_tinyimagenet(graph_model, degree):
    return vgg16_bn(graph_model, degree=degree, num_classes=200)

def generate(args, graph_model):
    if args.dataset == 'cifar10':
        return generate_cifar10(graph_model, args.degree)
    if args.dataset == 'tinyimagenet':
        return generate_tinyimagenet(graph_model, args.degree)
    else:
        raise Exception(f"Model type {args.model} with dataset {args.dataset} not implemented")