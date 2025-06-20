from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2
import argparse 

# # default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "pretrained/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map=0
# )

img_dir = "refcocog/val/val-49121.jpeg"
descibe = "the man with the bald head"
real_bbox = [119, 12, 423, 417]

def get_args_parser():
    parser = argparse.ArgumentParser('demo', add_help = False)
    parser.add_argument('--od',default = False,type = bool,
                        help = "Is object detection task?")
    return parser

def model_inference():
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "pretrained/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=1,
    )

    # default processor
    processor = AutoProcessor.from_pretrained("pretrained/Qwen2.5-VL-3B-Instruct")

    breakpoint()

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{img_dir}",
                },
                {"type": "text", "text": "Please provide the bounding box for the following description: {describe}"},
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
    output_text = model_inference()
    if args.od == True:
        # 目标检测任务用
        import re
        import cv2
        output = output_text[0]
        x = re.search('"bbox_2d":',output)
        end = x.end()
        output = output[end:]
        a = output.find("[")
        b = output.find("]")
        xmin,ymin,xmax,ymax = eval(output[a : b + 1])
        img = cv2.imread(img_dir)
        predict = cv2.rectangle(img.copy(),(xmin,ymin),(xmax,ymax),(0,0,255),3)
        cv2.imwrite("predict.png",predict)
        xmin,ymin,xmax,ymax = real_bbox
        label = cv2.rectangle(img.copy(),(xmin,ymin),(xmax,ymax),(0,0,255),3)
        cv2.imwrite("label.png",label)