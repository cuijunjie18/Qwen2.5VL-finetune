from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import argparse 
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

def get_args_parser():
    parser = argparse.ArgumentParser('Predict for small-coco', add_help = False)
    parser.add_argument('--model_dir',required = True,type = str,help = "Model_path")
    parser.add_argument('--image_path',required = True,type = str,help = "image_path")
    parser.add_argument('--use_lora',default = False,type = int,help = "If use LoRa-finetune model")
    parser.add_argument('--lora_model',default = None,type = str,help = "LoRa model path")
    return parser

def model_inference(args):
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processor
    processor = AutoProcessor.from_pretrained(args.model_dir)

    # 如果是lora微调推理
    if args.use_lora:
        print("=========================================================Use lora model!!!!")
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
        model = PeftModel.from_pretrained(model,args.lora_model,config = val_config)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{args.image_path}",
                },
                {"type": "text", "text": "Please describle this image."},
            ],
        }
    ]

    # Preparation for inference
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
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    output_text = model_inference(args)
    print(output_text)