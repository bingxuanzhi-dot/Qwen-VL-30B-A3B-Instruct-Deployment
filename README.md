# Qwen-VL-30B-Instruct Deployment
这是一个用于部署和推理 **Qwen-VL-30B-Instruct** 模型的项目，支持单图推理、多帧理解以及多轮对话。
## 🛠️ 环境安装 (Installation)

本项目建议使用 Conda 进行环境管理，Python 版本需为 3.10。

### 创建并激活虚拟环境

```bash
# 创建并激活虚拟环境
conda create -n qwen3_vl_30b_a3b_instruct python=3.10 -y
conda activate qwen3_vl_30b_a3b_instruct

#安装 PyTorch 及相关依赖
pip install torch torchvision torchaudio

# 安装 ModelScope、Accelerate 和 Qwen 工具包
pip install modelscope accelerate qwen-vl-utils

# 安装最新版 Transformers (推荐从源码安装以支持最新模型)：
pip install git+https://github.com/huggingface/transformers
```
