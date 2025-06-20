import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
import torch, gc
gc.collect()
torch.cuda.empty_cache()
# import deepspeed
# DS_CONFIG = "reference/ds_zero2_no_offload.json"
from datasets import Dataset
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel,get_peft_model_state_dict
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)
import swanlab
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 多GPU时可指定起始位置/编号



def load_and_convert_data(file_path):
    """加载并转换数据"""
    loaded_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            loaded_data.append(json.loads(line))

    # 将 loaded_data 转换为适合 Dataset 的格式
    dataset_dicts = []
    for item in loaded_data:
        user_content = item[0]['content']
        assistant_content = item[1]['content']

        # 提取图像和文本信息
        image_info = next((x for x in user_content if x['type'] == 'image'), None)
        text_info = next((x for x in user_content if x['type'] == 'text'), None)

        # 构造新的字典
        dataset_entry = {
            'role': 'user',
            'image_path': image_info['image'] if image_info else None,
            'question': text_info['text'] if text_info else None,
            'assistant_answer': assistant_content
        }
        
        dataset_dicts.append(dataset_entry)
    
    return dataset_dicts



def process_func_batch(examples):
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels, pixel_values, image_grid_thw = [], [], [], [], []
   
    for example in zip(examples["question"], examples["assistant_answer"], examples["image_path"]):
        input_content, output_content, file_path = example
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{file_path}",
                        # "resized_height": 280, # 为什么注释就错了?
                        # "resized_width": 280,
                    },
                    {"type": "text", "text": input_content},
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,  # 先不填充
            return_tensors="pt",
        )

        inputs_dict = {key: value.tolist() for key, value in inputs.items()}
        instruction_input_ids = inputs_dict['input_ids'][0]
        instruction_attention_mask = inputs_dict['attention_mask'][0]

        response = tokenizer(f"{output_content}", add_special_tokens=False)
        response_input_ids = response['input_ids']
        response_attention_mask = response['attention_mask']

        # 计算剩余可用长度给response
        remaining_length = MAX_LENGTH - len(instruction_input_ids) - 1  # 减去一个PAD token的空间

        if remaining_length < 0:
            # 如果指令部分已经超过最大长度，则需要截断指令部分
            truncation_length = len(instruction_input_ids) + remaining_length
            instruction_input_ids = instruction_input_ids[:truncation_length]
            instruction_attention_mask = instruction_attention_mask[:truncation_length]
            remaining_length = 0

        # 截断response部分以适应剩余空间
        current_input_ids = (
            instruction_input_ids + response_input_ids[:remaining_length] + [tokenizer.pad_token_id]
        )

        current_attention_mask = (
            instruction_attention_mask + response_attention_mask[:remaining_length] + [1]
        )
        current_labels = (
            [-100] * len(instruction_input_ids) +
            response_input_ids[:remaining_length] +
            [tokenizer.pad_token_id]
        )
        
        # 填充到MAX_LENGTH
        if len(current_input_ids) < MAX_LENGTH:
            current_input_ids += [tokenizer.pad_token_id] * (MAX_LENGTH - len(current_input_ids))
            current_attention_mask += [0] * (MAX_LENGTH - len(current_attention_mask))
            current_labels += [-100] * (MAX_LENGTH - len(current_labels))

        input_ids.append(current_input_ids)
        attention_mask.append(current_attention_mask)
        labels.append(current_labels)
        pixel_values.append(inputs_dict['pixel_values'])
        image_grid_thw.append(torch.tensor(inputs_dict['image_grid_thw']).squeeze(0))

    return {

        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": torch.tensor(pixel_values),
        "image_grid_thw": torch.stack(image_grid_thw)
    }

def predict(messages, model):
    # 准备推理  
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    # 将所有张量移动到指定的设备上
    for key, value in inputs.items():
        inputs[key] = value.to(device)
    
    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    del inputs
    
    return output_text[0]


# 在modelscope上下载Qwen2-VL模型到本地目录下
# model_dir = snapshot_download("Qwen/Qwen2-VL-2B-Instruct", cache_dir="./", revision="master")
model_dir = "pretrained/Qwen2.5-VL-3B-Instruct"

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels,use_fast=True)
# device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}   
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, 
    torch_dtype = torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map = "auto"
)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法/

# 处理数据集：读取json文件
# 分别加载 test 和 val 数据集
test_data_path = 'refcocog/data_test.jsonl'
val_data_path = 'refcocog/data_val.jsonl'

test_dataset_dicts = load_and_convert_data(test_data_path)
val_dataset_dicts = load_and_convert_data(val_data_path)

# 创建 Dataset 对象
test_tmp_dataset = Dataset.from_list(test_dataset_dicts)
val_tmp_dataset = Dataset.from_list(val_dataset_dicts)

indices = list(range(1000))
test_tmp_dataset = test_tmp_dataset.select(indices)
indices = list(range(50))
val_tmp_dataset = val_tmp_dataset.select(indices)

test_dataset = test_tmp_dataset.map(process_func_batch, batched=True,batch_size=1)
val_dataset = val_tmp_dataset.map(process_func_batch, batched=True, batch_size=1)

print("Test and Val Datasets have been created.")

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
# 转换模型
peft_model = get_peft_model(model, config)
peft_model.config.use_cache = False



# 配置训练参数
args = TrainingArguments(
    output_dir="./output_model/Qwen2.5-VL-3B",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=1,
    save_steps=50,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    # bf16=True,
    fp16=True,
    max_grad_norm=1.0, 
    deepspeed=None # 不使用deepspeed
)
        
# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen2.5-VL-finetune",
    experiment_name="qwen2.5-vl-coco2014",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct",
        "dataset": "https://huggingface.co/datasets/Kangheng/refcocog",
        "github": "https://github.com/datawhalechina/self-llm",
        "prompt": "Please provide the bounding box for the following descriptio: ",
        "train_data_number": len(test_dataset),
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=test_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)


# 开启模型训练
trainer.train()
trainer.save_model('./output2/Qwen2.5-VL-3B')
trainer.save_state()


# ====================测试模式===================
# 配置测试参数
val_config = LoraConfig(  
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)


# 获取测试模型
val_peft_model = PeftModel.from_pretrained(model, model_id="./output2/Qwen2.5-VL-3B/", config=val_config)


# 创建一个列表来保存所有需要的信息
results_to_save = []

# 同时创建test_image_list用于swanlab日志记录
test_image_list = []

for item in val_dataset:
    # 准备输入消息
    messages = [{
        "role": "user", 
        "content": [
            {
                "type": "image", 
                "image": item['image_path']
            },
            {
                "type": "text",
                "text": item['question']
            }
        ]}]
    
    # 获取模型预测
    response = predict(messages, val_peft_model)
    messages.append({"role": "assistant", "content": f"{response}"})
    
    # 打印或记录预测信息
    print(messages[-1])

    # 添加预测结果、原始答案和图片路径到结果列表中
    results_to_save.append({
        'image_path': item['image_path'],
        'question':item['question'],
        'original_answer': item['assistant_answer'],
        'predicted_answer': response,
    })

    # 同时添加到test_image_list用于SwanLab日志记录
    test_image_list.append(swanlab.Image(item['image_path'], caption=response))

# 定义保存文件的路径
output_file_path = './predictions_results.json'

# 将结果写入JSON文件
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(results_to_save, file, ensure_ascii=False, indent=4)

print(f"Results have been saved to {output_file_path}")
swanlab.init()
# 使用SwanLab记录预测结果
swanlab.log({"Prediction": test_image_list})

# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()
