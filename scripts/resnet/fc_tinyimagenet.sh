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

MODEL='resnet'
DATASET='tinyimagenet'

i=0
export CUDA_VISIBLE_DEVICES=$i

for DEPTH in 18 34 50 101 152; do
        exp_name=${MODEL}${DEPTH}_${DATASET}
        fname=../outputs/${exp_name}
        echo 'Running experiment '${exp_name}

        python -u main.py \
                --dataset=${DATASET} \
                --model=${MODEL} \
                --random_seed=${i} \
                --depth=${DEPTH} \
                --weight_decay=0.0001 \
                --num_workers=8 \
                --learning_rate=0.1 \
                --max_learning_rate=0.1 \
                --min_learning_rate=1e-05 \
                --epochs=300 \
                --batch_size=128 1> ${fname}.out 2> ${fname}.err
done
