import torch
from modelscope import Qwen3VLMoeForConditionalGeneration, AutoProcessor

# 1.初始化模型
model_id = "/share/public_models/qwen/Qwen3-VL/Qwen3-VL-30B-A3B-Instruct"
print(f"正在加载模型: {model_id}...")

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    model_id,
    dtype = torch.bfloat16,
    device_map = "auto",
    trust_remote_code = True
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True)

# 2.初始化对话历史
image_path = "/home/zhibingxuan/qwen3_vl_30b_a3b_instruct/multi_turn/pic.jpg"

# messages 是一个对话列表，存储整个对话的"记忆", 
messages = [
    {
        "role":"user",
        "content":[
            {
                "type": "image",
                "image": image_path
            },
            {
                "type": "text",
                "text": "请描述一下这张图片。"
            }
        ]
    }
]
print("="*20)
print("开始多轮对话 (输入 'exit' 或 'quit' 退出)")
print("="*20)

# 3.进入多轮对话循环
first_turn = True # 标记是否是第一轮

while True:
    if not first_turn:
        user_input = input("\nUser:")
        if user_input.lower() in ["exit", "quit"]:
            print("退出对话")
            break

        # 将用户的新问题追加到历史记录中
        messages.append({
            "role": "user",
            "content":[
                {
                    "type": "text",
                    "text": user_input
                }
            ]
        })
    else:
        print(f"User: {messages[0]['content'][1]['text']}")
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_dict = True,
        return_tensors = "pt"
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens = 512,
        do_sample = True,
        temperature = 0.7,
        top_p = 0.9
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens = True, 
        clean_up_tokenization_spaces = False
    )[0]

    print(f"Assistant: {output_text}")

    # 把 AI 的回答也加入"记忆"列表
    messages.append({   
        "role": "assistant",
        "content":[
            {
                "type": "text",
                "text": output_text
            }
        ]
    })

    first_turn = False