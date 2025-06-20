#!/bin/bash

export CUDA_VISIBLE_DEVICES="0" # 设置可用的gpu列表
log_name="infer-small-coco"
model_dir="pretrained/Qwen2.5-VL-3B-Instruct"
image_path="small-coco/train/train-000000156.png"
use_lora=1
lora_model="output/Qwen2.5-VL-3B/checkpoint-40"
# lora_model="None"


args="
    --model_dir ${model_dir} \
    --image_path ${image_path} \
    --use_lora ${use_lora} \
    --lora_model ${lora_model}
"

uv run utils/predict_for_small-coco.py ${args} \
# 2>&1 | tee logs/${log_name}$(date '+%Y-%m-%d_%H-%M-%S').log