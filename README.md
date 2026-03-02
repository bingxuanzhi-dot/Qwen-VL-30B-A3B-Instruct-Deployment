# Qwen-VL-30B-Instruct Deployment 🚀
这是一个用于部署和推理 **Qwen-VL-30B-Instruct** 模型的项目，支持单图推理、多帧理解以及多轮对话。
## 🛠️ 环境安装 (Installation)

本项目建议使用 Conda 进行环境管理，Python 版本需为 3.10。

### 创建并激活虚拟环境

```bash
conda create -n qwen3_vl_30b_a3b_instruct python=3.10 -y
conda activate qwen3_vl_30b_a3b_instruct

pip install torch torchvision torchaudio

pip install modelscope accelerate qwen-vl-utils

pip install git+https://github.com/huggingface/transformers
```
