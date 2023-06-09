"""
This code is adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
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

from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from model_factory.layers.convolutional_layers.sparse_graph_conv2d import SparseGraphConv2d

def conv3x3(graph_model, degree: int, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """ 3x3 convolution with padding """

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, \
                     groups=groups, bias=False, dilation=dilation) if graph_model == "fc" \
                else SparseGraphConv2d(graph_model, in_planes, out_planes, degree=degree, kernel_size=3, \
                                       stride=stride, padding=dilation, groups=groups, dilation=dilation)

def conv1x1(graph_model, degree: int, in_planes: int, out_planes: int, stride: int = 1):
    """ 1x1 convolution """

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False) if graph_model == 'fc' \
                else SparseGraphConv2d(graph_model, in_planes, out_planes, degree=degree, kernel_size=1, stride=stride)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        graph_model, 
        degree: int,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(graph_model, degree, inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(graph_model, degree, planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        graph_model,
        degree: int,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(graph_model, degree, inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(graph_model, degree, width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(graph_model, degree, width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        graph_model,
        degree: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(graph_model, degree, block, 64, layers[0])
        self.layer2 = self._make_layer(graph_model, degree, block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(graph_model, degree, block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(graph_model, degree, block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        graph_model,
        degree: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(graph_model, degree, self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                graph_model, degree, self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    graph_model,
                    degree,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def resnet18(graph_model, degree: int, num_classes: int) -> ResNet:
    """ Returns ResNet 18. """

    return ResNet(graph_model, degree, BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(graph_model, degree: int, num_classes: int) -> ResNet:
    """ Returns ResNet 34. """

    return ResNet(graph_model, degree, BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(graph_model, degree: int, num_classes: int) -> ResNet:
    """ Returns ResNet 50. """

    return ResNet(graph_model, degree, Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(graph_model, degree: int, num_classes: int) -> ResNet:
    """ Returns ResNet 101. """
    
    return ResNet(graph_model, degree, Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(graph_model, degree: int, num_classes: int) -> ResNet:
    """ Returns ResNet 152. """

    return ResNet(graph_model, degree, Bottleneck, [3, 8, 36, 3], num_classes)

def generate_tinyimagenet(args, graph_model):
    """ Returns model for Tiny ImageNet corresponding to args and graph_model. """

    if (args.depth == 18):
        return resnet18(graph_model, degree=args.degree, num_classes=200)
    if (args.depth == 34):
        return resnet34(graph_model, degree=args.degree, num_classes=200)
    if (args.depth == 50):
        return resnet50(graph_model, degree=args.degree, num_classes=200)
    if (args.depth == 101):
        return resnet101(graph_model, degree=args.degree, num_classes=200)
    if (args.depth == 152):
        return resnet152(graph_model, degree=args.degree, num_classes=200)

    raise NotImplementedError(f"Model type {args.model} with depth {args.depth} not implemented")
    
def generate_cifar10(args, graph_model):
    """ Returns model for CIFAR-10 corresponding to args and graph_model. """

    if (args.depth == 18):
        return resnet18(graph_model, degree=args.degree, num_classes=10)
    if (args.depth == 34):
        return resnet34(graph_model, degree=args.degree, num_classes=10)
    if (args.depth == 50):
        return resnet50(graph_model, degree=args.degree, num_classes=10)
    if (args.depth == 101):
        return resnet101(graph_model, degree=args.degree, num_classes=10)
    if (args.depth == 152):
        return resnet152(graph_model, degree=args.degree, num_classes=10)

    raise NotImplementedError(f"Model type {args.model} with depth {args.depth} not implemented")

def generate(args, graph_model):
    """ Returns model corresponding to args and graph_model. """

    if args.dataset == 'tinyimagenet':
        return generate_tinyimagenet(args, graph_model)
    if args.dataset == 'cifar10':
        return generate_cifar10(args, graph_model)

    raise NotImplementedError(f"Model type {args.model} with dataset {args.dataset} not implemented")
    