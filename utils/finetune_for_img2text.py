import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,3" # set the cuda index

import json
import torch
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from transformers import(
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

import swanlab
from swanlab.integration.transformers import SwanLabCallback # 用于记录训练情况

from peft import LoraConfig, TaskType, get_peft_model, PeftModel # LoRa微调

import argparse

def get_parser()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Train image to text in small-coco dataset', add_help = False)
    parser.add_argument("--model_dir",required = True,type = str,help = "Model dir")
    parser.add_argument("--dataset",required = True,type = str,help = "dataset json file(train and test)")
    parser.add_argument("--batch_size",required = True,type = int,help = "batch size for training.")
    return parser

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    # "resized_height": 280,
                    # "resized_width": 280,
                },
                {"type": "text", "text": "Please describle this image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本

    # breakpoint()

    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)


    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  #由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}

# 解析命令参数
parser = get_parser()
args = parser.parse_args()    

# 加载模型
model_dir = args.model_dir
tokenizer = AutoTokenizer.from_pretrained(model_dir,use_fast = True)
processor = AutoProcessor.from_pretrained(model_dir,use_fast = True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype = torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map = "auto"
)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 加载数据
indices = list(range(1000))
train_data_json = args.dataset
train_ds = Dataset.from_json(train_data_json)
train_ds = train_ds.select(indices) # 仅选择前1000个来训练

print("========================================================")
print("Dataset build....")
train_dataset = train_ds.map(process_func)
print("Finished!")


# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,   # 训练模式
    r=64,                   # Lora 秩
    lora_alpha=16,          # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,      # Dropout 比例
    bias="none",
)

# 获取LoRA模型
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()  # 打印可训练参数信息

# 配置训练参数
train_args = TrainingArguments(
    output_dir="./output/Qwen2.5-VL-3B",
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    # logging_steps=10,
    # logging_first_step=5,
    num_train_epochs=10,
    # save_steps=100,
    learning_rate=1e-4,
    # save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# # 设置SwanLab回调
# swanlab_callback = SwanLabCallback(
#     project="Qwen2.5-VL-3B-finetune",
#     experiment_name="qwen2-vl-coco",
#     config={
#         "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct",
#         "dataset": "https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart",
#         "github": "https://github.com/datawhalechina/self-llm",
#         "prompt": "COCO Yes: ",
#         "train_data_number": len(train_ds),
#         "lora_rank": 64,
#         "lora_alpha": 16,
#         "lora_dropout": 0.1,
#     },
# )

# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=train_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    # callbacks=[swanlab_callback],
)

# 开启模型训练
trainer.train()

# 结束
# swanlab.finish()
