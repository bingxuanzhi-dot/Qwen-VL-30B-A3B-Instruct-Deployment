import torch
from modelscope import Qwen3VLMoeForConditionalGeneration, AutoProcessor

model_id = "Qwen3-VL/Qwen3-VL-30B-A3B-Instruct"
print(f"正在加载模型: {model_id}...")

# 1.加载模型
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    model_id,
    dtype = torch.bfloat16,
    device_map = "auto",
    trust_remote_code = True,
    # attn_implementation = "flash_attention_2",
)

# 2.加载处理器
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True)

# 3.准备输入设备
image_path_1 = "qwen3_vl_30b_a3b_instruct/double/frame_00000.jpg"
image_path_2 = "qwen3_vl_30b_a3b_instruct/double/frame_00015.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path_1
            },
            {
                "type": "image",
                "image": image_path_2
            },
            {
                "type": "text",
                "text": "请结合这两张图片的内容，描述从第一张图片到第二张图片的动作行为，并分析为什么要进行这个动作。",
            }
        ]
    }
]

# 4.预处理输入
inputs = processor.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_dict = True,
    return_tensors = "pt"
)

inputs = inputs.to(model.device)

# 5.开始推理
print("开始生成...")
generated_ids = model.generate(
    **inputs,
    max_new_tokens = 256,
    do_sample = True,
    temperature = 0.7
)

# 6.解码输出
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens = True,
    clean_up_tokenization_spaces = False,
)

print("-" * 20)
print("模型回复:")
print(output_text[0])
