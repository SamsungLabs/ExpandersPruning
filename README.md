<!-- Copyright (c) 2023 Samsung Electronics Co., Ltd.

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
For conditions of distribution and use, see the accompanying LICENSE.md file. -->

# Data-Free Model Pruning at Initialization via Expanders

This respository contains the code and experiments for the paper [Data-Free Model Pruning at Initialization via Expanders](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Stewart_Data-Free_Model_Pruning_at_Initialization_via_Expanders_CVPRW_2023_paper.pdf), published at the Efficient Deep Learning for Computer Vision CVPR Workshop, 2023.

Authors: James Stewart, Umberto Michieli, and Mete Ozay.

## Requirements and installation

A list of requirements for this project can be found in ```environment.yml```. 

To create and activate a conda environment from these requirements, run ```conda env create -f environment.yml``` and ```conda activate config_model_env```.

## Architectures

We provide support for fully-connected and sparsified versions of VGG 16 and ResNet (depth 18, 32, 50, 101, and 152). Their implementations can be found in ```src/model_factory/models```.

Implementations of other models should be placed here, and added to ```src/model_factory/model_factory.py```. The list of supported models found in ```src/utils/config.py``` should also be updated.

## Datasets

We provide support for the CIFAR-10 and Tiny ImageNet datasets. In the former case, the data will be downloaded to ```data/```; in the latter case, the data should be placed here. The Tiny ImageNet dataset can be downloaded from ```http://cs231n.stanford.edu/tiny-imagenet-200.zip```.
After downloading it, run the script ```data/prepare_tiny_imagenet.py``` to prepare the data.

To add a new dataset, a custom data loader should be implemented according to the interface specified by ```src/data_loader_factory/data_loaders/data_loader.py```. The list of supported datasets found in ```src/utils/config.py``` should also be updated.

## Graph models

We implement sparse convolutional and linear layers according to three different random graph models: a random graph (Random), a random regular graph (RReg), and a random left-regular graph (XNet). The implementations of these graph masked layers can be found in ```src/model_factory/layers```.

New graph models should be implemented in ```src/graph_factory/graph_utils.py``` and added to ```src/graph_factory/graph_factory.py```. The architecture to be sparsified according to the new graph model may then be added to the model factory, as per the above instructions. 

## Getting started

After installing the requirements, running the command ```main.py --model='vgg16bn' --dataset='cifar10'``` from ```src/``` will train and test a fully-connected VGG 16 model on the CIFAR-10 dataset for 300 epochs using the defauly hyperparameters. We can sparsify this model using random graphs of average degree 3 by running ```main.py --model='vgg16bn_random' --degree=3 --dataset='cifar10'```. For all other options and information on the default hyperparameters, see ```src/arg_parser/arg_parser.py```.

## Experiments

To reproduce the results for VGG 16 and ResNet on both CIFAR-10 and Tiny ImageNet, we provide a number of scripts located in ```scripts/```. From the ```src/``` folder, simply run ```bash ../scripts/[script name]``` to execute each of these. The console logs will be written to ```outputs/```.

## License

This work is released under the Attribution-NonCommercial-ShareAlike 4.0 International license. Parts of this work are adapted from [PyTorch](https://github.com/pytorch/), which is released under the 3-Clause BSD license.

## Citation

```
@InProceedings{stewart2022cvpr,
    author    = {Stewart, James and Michieli, Umberto and Ozay, Mete},
    title     = {Data-Free Model Pruning at Initialization via Expanders},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023}
}
```
