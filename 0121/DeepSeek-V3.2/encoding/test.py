import json
import os
from tqdm import tqdm
import copy
# 确保 encoding_dsv32.py 在你的当前目录下或 Python 路径中
from encoding_dsv32 import encode_messages

def fix_data_structure(messages, tools):
    """
    深度修复数据结构，确保符合 encoding_dsv32 的 OpenAI 格式要求
    """
    # 1. 修复根节点的 tools 格式
    fixed_tools = []
    if tools:
        for t in tools:
            if isinstance(t, dict) and "function" in t:
                fixed_tools.append(t)
            elif isinstance(t, dict) and "name" in t:
                fixed_tools.append({"type": "function", "function": t})
            else:
                fixed_tools.append(t)
    
    # 2. 深度拷贝 messages 以免修改原始数据，并修复其中的 tool_calls
    fixed_messages = copy.deepcopy(messages)
    for msg in fixed_messages:
        # 如果 assistant 消息中有 tool_calls
        if "tool_calls" in msg and msg["tool_calls"]:
            new_tool_calls = []
            for tc in msg["tool_calls"]:
                # 如果 tool_calls 里缺少 'function' 层级，手动补齐
                if isinstance(tc, dict) and "function" not in tc:
                    new_tool_calls.append({
                        "id": tc.get("id", "call_default"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": tc.get("arguments")
                        }
                    })
                else:
                    new_tool_calls.append(tc)
            msg["tool_calls"] = new_tool_calls

    # 3. 将修复后的 tools 注入第一条消息
    if fixed_tools and len(fixed_messages) > 0:
        fixed_messages[0]["tools"] = fixed_tools
        
    return fixed_messages

def main():
    # 原始输入路径
    input_path = "/gfs/space/chatrl/users/hxh/code/agent_scripts/data/output/250305-250529_math_combined-41649_no_repeat_70_turn1_deepseekv3.2.jsonl"
    # Slime 专用输出路径
    output_path = "/gfs/space/chatrl/users/djy/0121/sft_code/math_with_tools_slime_format.jsonl"

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # DeepSeek V3.2 的关键切分标志
    SPLIT_MARKER = "<｜Assistant｜>"

    # 统计行数以便显示进度
    print("正在扫描原始文件...")
    try:
        total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Slime 格式转换中"):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                
                # 1. 修复数据结构以适配 encoding 函数
                fixed_messages = fix_data_structure(
                    data.get("messages", []), 
                    data.get("tools", [])
                )

                # 2. 编码配置：包含思维链，添加 BOS
                encode_config = dict(
                    thinking_mode="thinking", 
                    drop_thinking=False, 
                    add_default_bos_token=True
                )

                # 3. 获取完整的 DeepSeek 格式化文本
                full_text = encode_messages(fixed_messages, **encode_config)
                
                # 4. 核心：切分为 Slime 框架需要的 prompt 和 label 字段
                if SPLIT_MARKER in full_text:
                    parts = full_text.split(SPLIT_MARKER)
                    
                    # prompt：从 BOS 开始到 Assistant 标签结束（含标签）
                    # 它是模型的“输入”，诱导模型开始在标签后生成内容
                    prompt_field = parts[0] + SPLIT_MARKER
                    
                    # label：Assistant 标签之后的所有内容（含思考过程、答案和 EOS）
                    # 它是模型的“目标”，通过计算这部分的 Loss 来实现 SFT
                    label_field = SPLIT_MARKER.join(parts[1:]) 
                    
                    # 5. 写入 JSONL，每行包含两个字段
                    output_obj = {
                        "prompt": prompt_field,
                        "label": label_field
                    }
                    f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                else:
                    tqdm.write(f"跳过：在编码文本中未找到分割符 {SPLIT_MARKER}")
                
            except Exception as e:
                tqdm.write(f"处理失败跳过一条: {type(e).__name__}: {str(e)}")

    print(f"\n✅ 转换完成！")
    print(f"Slime 训练数据已保存至: {output_path}")

if __name__ == "__main__":
    main()