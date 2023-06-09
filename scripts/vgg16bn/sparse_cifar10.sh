# Copyright (c) 2023 Samsung Electronics Co., Ltd.
#
# Author(s):
# James Stewart (j2.stewart@samsung.com; james1995stewart@gmail.com)
# Umberto Michieli (u.michieli@samsung.com)
# Mete Ozay (m.ozay@samsung.com)
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# For conditions of distribution and use, see the accompanying LICENSE.md file.

DATASET='cifar10'

i=0
export CUDA_VISIBLE_DEVICES=$i

for MODEL in 'vgg16bn_rreg' 'vgg16bn_xnet' 'vgg16bn_random'; do

        for DEGREE in 3 15 30 60; do

                exp_name=${MODEL}_degree_${DEGREE}_${DATASET}
                fname=../outputs/${exp_name}
                echo 'Running experiment '${exp_name}

                python -u main.py \
                        --dataset=${DATASET} \
                        --model=${MODEL} \
                        --degree=${DEGREE} \
                        --batch_size=128 \
                        --criterion='crossentropy' \
                        --weight_decay=0.0005 \
                        --learning_rate=0.05 \
                        --min_learning_rate=0.0005 \
                        --max_learning_rate=0.05 \
                        --epochs=300 \
                        --num_workers=2 \
                        --optimizer='sgd' \
                        --nesterov=True \
                        --random_seed=${i} \
                        1> ${fname}.out 2> ${fname}.err
        done

done
