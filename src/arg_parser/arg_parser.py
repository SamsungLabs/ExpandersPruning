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

import argparse

from utils.config import models
from utils.config import datasets
from utils.config import criterions
from utils.config import optimizers
from utils.config import lr_schedulers

def create_parser():
    """ Creates argparser. """

    parser = argparse.ArgumentParser(description="Configuration model")

    parser.add_argument("--model", required=True, choices=models, \
                        help="Type of architecture to be generated")
    parser.add_argument("--dataset", required=True, choices=datasets, help="Dataset")
    parser.add_argument("--checkpoint", required=False, help="Checkpoint for model weights")
    parser.add_argument("--cuda", required=False, default=True, help="Enable CUDA")
    parser.add_argument("--gpu", required=False, default=0, help="GPU ID")
    parser.add_argument("--batch_size", required=False, default=128, type=int, help="Batch size")
    parser.add_argument("--num_workers", required=False, default=15, type=int, \
                        help="Number of workers")
    parser.add_argument("--random_seed", required=False, default=123, type=int, help="Random seed")
    parser.add_argument("--criterion", required=False, choices=criterions, default="crossentropy", \
                        help="Loss function")
    parser.add_argument("--optimizer", required=False, choices=optimizers, default="sgd", \
                        help="Optimizer for training")
    parser.add_argument("--epochs", required=False, default=300, type=int, \
                        help="Number of epochs to train for")
    parser.add_argument("--momentum", required=False, default=0.9, help="Momentum")
    parser.add_argument("--nesterov", required=False, default=True, help="Nesterov momentum")
    parser.add_argument("--weight_decay", required=False, default=0.0005, type=float, \
                        help="Weight decay")
    parser.add_argument("--learning_rate", required=False, default=0.05, type=float, \
                        help="Learning rate")
    parser.add_argument("--lr_scheduler", required=False, choices=lr_schedulers, default="steplr", \
                        help="Learning rate scheduler")
    parser.add_argument("--min_learning_rate", required=False, default=0.0005, type=float, \
                        help="Minimum learning rate")
    parser.add_argument("--max_learning_rate", required=False, default=1, type=float, \
                        help="Maximum learning rate")
    parser.add_argument("--lr_decay_rate", required=False, default=0.5, type=float, \
                        help="Decay rate for learning rate scheduler")
    parser.add_argument("--lr_decay_step", required=False, default=30, type=int, \
                        help="Step size for learning rate scheduler")
    parser.add_argument("--tensorboard", required=False, default=True, \
                        help="Log results on tensorboard")
    parser.add_argument("--degree", required=False, default=-1, type=int, \
                        help="Average degree of sparse layer graph")
    parser.add_argument("--depth", required=False, default=-1, type=int, help="Depth of network")

    return parser
