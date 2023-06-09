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
import random

import torch
import torchvision
from torchvision import transforms

from utils.config import tinyimagenet_train_path
from utils.config import tinyimagenet_test_path
from data_loader_factory.data_loaders.data_loader import DataLoader

class TinyImageNetCroppedLoader(DataLoader):
    """ Implementation of DataLoader interface for Tiny ImageNet (cropped). """

    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.random_seed = args.random_seed

        torch.random.manual_seed(1)
        random.seed(1)

        transform = transforms.Compose([transforms.RandomCrop(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                                             std=[0.229, 0.224, 0.225])])

        self.train_set = torchvision.datasets.ImageFolder(root=tinyimagenet_train_path, \
                                                          transform=transform)
        self.test_set = torchvision.datasets.ImageFolder(root=tinyimagenet_test_path, \
                                                         transform=transform)

    def get_train_loader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, \
                                           shuffle=True, num_workers=60)

    def get_val_loader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, \
                                           shuffle=False, num_workers=50)

    def get_test_loader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, \
                                           shuffle=False, num_workers=50)
    