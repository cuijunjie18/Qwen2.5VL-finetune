# Qwen2.5VL目标检测微调

## 背景

做实验，赶进度.

## 日志

- 2025-6-16
  - 学习了huggingface开源数据集的使用.
  - 完成数据集Qwen2.5VL格式的处理.

- 2025-6-17
  - 目标检测任务训练跑通，但是存在问题： 为什么关闭构造的message中关闭resize后Dataset构造报错.
  - 训练问题： 无法开启deepspeed训练，报错，初步判断是版本问题.
  - 学会设置环境变量去指定多卡训练.

## 收获

- datasets库的load_dataset使用
  ```py
  # 从本地读取
  ds = load_dataset("refcocog")
  
  # 在线下载： 使用huggingface镜像站
  os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
  ds = load_dataset("Kangheng/refcocog")
  ```

- json与jsonl的纠缠
  分别实现的不同处理见
  ```shell
  my_data.ipynb # jsonl处理
  data_process_for_coco.ipynb # json处理
  ```

- 设置环境变量修改可用gpu范围
  ```py
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
  ```
  **之后cuda:0即实际的cuda:1了，cuda:1即实际的cuda:3了.**   
  **注：上面的设置环境变量要在torch导入前.**