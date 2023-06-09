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

import os

models = ["vgg16bn", "vgg16bn_rreg", "vgg16bn_xnet", "vgg16bn_random",
          "resnet", "resnet_rreg", "resnet_xnet", "resnet_random"]

datasets = ["cifar10", "tinyimagenet"]

criterions = ["crossentropy"]
optimizers = ["sgd"]
lr_schedulers = ["steplr"]

DATA_DIR = "../data/"

tinyimagenet_train_path = os.path.join(DATA_DIR, "tiny-imagenet-200/train")
tinyimagenet_test_path = os.path.join(DATA_DIR, "tiny-imagenet-200/val/images")

TENSORBOARD_DIR = "../runs/"

MODEL_SAVE_DIR = "../trained_models/"

GRAPH_SAVE_DIR = "../graphs/"
