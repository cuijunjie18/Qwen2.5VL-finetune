# Qwen2.5VL目标检测微调

## 背景

做实验，赶进度.

## 日志

- 2025-6-17
  - 学习了huggingface开源数据集的使用.
  - 完成数据集Qwen2.5VL格式的处理.

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