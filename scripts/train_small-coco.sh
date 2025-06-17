#!/bin/bash

export CUDA_VISIBLE_DEVICES="3" # 设置训练的gpu
log_name="finetune-small-coco"
model_dir="pretrained/Qwen2.5-VL-3B-Instruct"
dataset="small-coco/data_train.json"
batch_size=64

args="
    --model_dir ${model_dir} \
    --dataset ${dataset} \
    --batch_size ${batch_size}
"

uv run utils/train_for_img2text.py ${args} \
# 2>&1 | tee logs/${log_name}$(date '+%Y-%m-%d_%H-%M-%S').log