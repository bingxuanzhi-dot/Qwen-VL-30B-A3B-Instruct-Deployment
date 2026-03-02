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

## 🚀 快速开始 (Quick Start)
在使用前，请确保已经下载了模型权重(https://www.modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct/summary)
1. 单图推理
```bash
python single_frame/run_qwen3_single_frame.py
```
2. 多帧理解
```bash
python multi_frame/run_qwen3_multi_frame.py
```
3. 多轮对话
```bash
python multi_turn/run_qwen3_multi_turn.py
```
## 👏 致谢 (Acknowledgement)

*   **Qwen-VL**: 本项目的核心模型来自阿里云 Qwen 团队，感谢他们的开源贡献。
*   **Transformers**: 本项目推理脚本基于 Hugging Face Transformers 库构建。

如有侵权或模型使用问题，请参考原模型的开源协议。
