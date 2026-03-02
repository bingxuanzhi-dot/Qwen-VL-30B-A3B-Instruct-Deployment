import torch
from modelscope import Qwen3VLMoeForConditionalGeneration,AutoProcessor

model_id = "Qwen3-VL/Qwen3-VL-30B-A3B-Instruct"
print(f"正在加载模型: {model_id}...")

# 1.加载模型
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    # attn_implementation = "flash_attention_2",
    device_map = "auto", # 自动决定把模型放在哪张显卡
    trust_remote_code = True
)

# 2.加载处理器
"""
这是配套的处理器，包含：
1. Tokenizer: 把文字变成 token 的数字序列
2. Image Processor: 调整图片大小并进行归一化, 最终得到张量
"""
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True)

# 3.准备输入设备
messages = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "image":"qwen3_vl_30b_a3b_instruct/single/pic.jpg"
            },
            {
                "type":"text",
                "text":"请相似描述一下这张图片的内容",
            }
        ]
    }
]

# 4.预处理输入
inputs = processor.apply_chat_template(
    messages,
    tokenize = True, #是否把文字转换为数字
    add_generation_prompt = True, #是否自动添加"<|im_start|>assistant"这样的引导符
    return_dict = True, #是否返回字典格式
    return_tensors = "pt" #返回 Pytorch 格式的张量
)

inputs = inputs.to(model.device)

# 5.开始推理
print("开始生成...")
generated_ids = model.generate(
    **inputs,
    max_new_tokens = 128, #最大生成长度为 128
    do_sample = True, #开启随机采样
    temperature = 0.7
) # generated_ids 是模型输出的一串数字，包含[输入内容 + 输出内容]

# 6.解码输出
# 把输入内容从结果里切掉，只保留模型新生成的回答
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# 把剩下的数字（Token IDs）翻译回人类文字。
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens = True, # 去掉 <|endoftext|> 这种特殊符号，不显示出来。
    clean_up_tokenization_spaces = False,
)

print("-"*20)
print("模型回复:")
print(output_text[0])