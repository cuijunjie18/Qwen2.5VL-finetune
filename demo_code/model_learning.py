import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer,AutoProcessor
from qwen_vl_utils import process_vision_info

model_dir = "pretrained/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_dir)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "small-coco/train/train-000000047.png",
            },
            {"type": "text", "text": "Please describle this image."},
        ],
    }
]

breakpoint()

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)

model_inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)


generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)