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
