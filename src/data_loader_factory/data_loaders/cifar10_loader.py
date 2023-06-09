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
import torchvision
from torchvision import transforms

from data_loader_factory.data_loaders.data_loader import DataLoader
from utils.config import DATA_DIR

class Cifar10Loader(DataLoader):
    """ Implementation of DataLoader interface for CIFAR-10. """

    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.random_seed = args.random_seed

        cifar10_mean = [x/255.0 for x in [125.3, 123.0, 113.9]]
        cifar10_std = [x/255.0 for x in [63.0, 62.1, 66.7]]

        train_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=cifar10_mean, \
                                                                   std=cifar10_std),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip()])

        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=cifar10_mean, \
                                                                  std=cifar10_std)])

        self.train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, \
                                                      download=True, transform=train_transform)
        self.test_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, \
                                                     download=True, transform=test_transform)

    def get_train_loader(self):
        return torch.utils.data.DataLoader(self.train_set,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True,
                                           pin_memory=True)

    def get_val_loader(self):
        return torch.utils.data.DataLoader(self.test_set,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True,
                                           pin_memory=True)

    def get_test_loader(self):
        return torch.utils.data.DataLoader(self.test_set,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True,
                                           pin_memory=True)
