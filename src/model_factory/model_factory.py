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

from model_factory.models.vgg import vgg16bn
from model_factory.models.resnet import resnet

def generate_new_model(args):
    """ Returns the relevant model given args. """

    if args.model == "vgg16bn":
        return vgg16bn.generate(args, "fc")
    if args.model == "vgg16bn_rreg":
        return vgg16bn.generate(args, "rreg")
    if args.model == "vgg16bn_xnet":
        return vgg16bn.generate(args, "xnet")
    if args.model == "vgg16bn_random":
        return vgg16bn.generate(args, "random")
    if args.model == "resnet":
        return resnet.generate(args, "fc")
    if args.model == "resnet_rreg":
        return resnet.generate(args, "rreg")
    if args.model == "resnet_xnet":
        return resnet.generate(args, "xnet")
    if args.model == "resnet_random":
        return resnet.generate(args, "random")

    raise NotImplementedError(f"Model type {args.model} not implemented")

def generate_model(args):
    """ Returns the relevant model given args.
        Loads weights if present. """

    model = generate_new_model(args)

    if args.checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])

    return model
