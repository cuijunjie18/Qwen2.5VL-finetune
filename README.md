# Qwen2.5VL目标检测微调

## 背景

做实验，赶进度.

## 目录说明

- link-file： 链接文件均为数据集或模型
- utils/ ： 训练、推理代码
- scripts/ ： 训练、推理执行脚本
- pyproject.toml： uv环境依赖

## 日志

- 2025-6-16
  - 学习了huggingface开源数据集的使用.
  - 完成数据集Qwen2.5VL格式的处理.

- 2025-6-17
  - 目标检测任务训练跑通，但是存在问题： 为什么关闭构造的message中关闭resize后Dataset构造报错.
  - 训练问题： 无法开启deepspeed训练，报错，初步判断是版本问题.
  - 学会设置环境变量去指定多卡训练.
  - 成功微调coco的image to text任务.

## 收获

- datasets库的load_dataset使用
  ```py
  # 从本地读取
  ds = load_dataset("refcocog")
  
  # 在线下载： 使用huggingface镜像站
  os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
  ds = load_dataset("Kangheng/refcocog")
  ```
  <br>
- json与jsonl的纠缠
  分别实现的不同处理见
  ```shell
  my_data.ipynb # jsonl处理
  data_process_for_coco.ipynb # json处理
  ```
  <br>
- 设置环境变量修改可用gpu范围
  ```py
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
  ```
  **之后cuda:0即实际的cuda:1了，cuda:1即实际的cuda:3了.**   
  **注：上面的设置环境变量要在torch导入前.**
  <br>
- bash/zsh脚本的根路径是执行这个脚本命令的路径
  如当前在下面路径
  ```shell
  /data_all/cjj_node/Qwen2.5VL_finetune
  ```

  脚本路径及内容
  ```shell
  # 脚本路径
  /data_all/cjj_node/Qwen2.5VL_finetune/scripts/predict_small-coco.sh

  # 脚本内容
  #!/bin/bash
  echo $(pwd)
  ```

  执行脚本
  ```shell
  # 当前/data_all/cjj_node/Qwen2.5VL_finetune
  bash scripts/predict_small-coco.sh

  # 输出
  /data_all/cjj_node/Qwen2.5VL_finetune
  ```
  <br>
- python 脚本的路径机制同bash/zsh脚本
- 脚本输出内容重定向到日志
  ```shell
  uv run utils/train_for_img2text.py ${args} \
  2>&1 | tee logs/${log_name}$(date '+%Y-%m-%d_%H-%M-%S').log
  ```
  这个重定向会在一定程度上影响输出样式.要注意。