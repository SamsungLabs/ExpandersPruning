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

from data_loader_factory.data_loaders.cifar10_loader import Cifar10Loader
from data_loader_factory.data_loaders.tinyimagenet_loader import TinyImageNetLoader
from data_loader_factory.data_loaders.tinyimagenet_cropped_loader import TinyImageNetCroppedLoader

def generate_data_loader(args):
    """ Returns the relevant data loader given args. """

    if args.dataset == "cifar10":
        return Cifar10Loader(args)
    if args.dataset == "tinyimagenet":
        if args.model.startswith("vgg16bn"):
            return TinyImageNetCroppedLoader(args)
        return TinyImageNetLoader(args)

    raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    