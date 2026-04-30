import json
import os
from tqdm import tqdm
import copy

from encoding_dsv32 import encode_messages


def fix_data_structure(messages, tools):
    """
    深度修复数据结构，确保符合 encoding_dsv32 的 OpenAI 格式要求
    """
    fixed_tools = []
    if tools:
        for t in tools:
            if isinstance(t, dict) and "function" in t:
                fixed_tools.append(t)
            elif isinstance(t, dict) and "name" in t:
                fixed_tools.append({"type": "function", "function": t})
            else:
                fixed_tools.append(t)
    
    fixed_messages = copy.deepcopy(messages)
    for msg in fixed_messages:
        if "tool_calls" in msg and msg["tool_calls"]:
            new_tool_calls = []
            for tc in msg["tool_calls"]:
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

    if fixed_tools and len(fixed_messages) > 0:
        fixed_messages[0]["tools"] = fixed_tools
        
    return fixed_messages

def main():
    # --- 路径配置 ---
    input_path = "/gfs/space/chatrl/users/hxh/code/agent_scripts/data/output/250305-250529_math_combined-41649_no_repeat_70_turn1_deepseekv3.2.jsonl"
    output_path = "/gfs/space/chatrl/users/djy/0121/sft_code/math_with_tools_slime_format_fixed.jsonl"

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    SPLIT_MARKER = "<｜Assistant｜>"

    print("正在扫描原始文件并转换格式...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="适配 Slime 格式"):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                
                # 1. 修复数据结构
                fixed_messages = fix_data_structure(
                    data.get("messages", []), 
                    data.get("tools", [])
                )

                # 2. 编码配置
                encode_config = dict(
                    thinking_mode="thinking", 
                    drop_thinking=False, 
                    add_default_bos_token=True
                )

                # 3. 获取完整的文本
                full_text = encode_messages(fixed_messages, **encode_config)
                
                # 4. 切分为 Slime 需要的格式
                if SPLIT_MARKER in full_text:
                    parts = full_text.split(SPLIT_MARKER)
                    prompt_content = parts[0] + SPLIT_MARKER
                    label_content = SPLIT_MARKER.join(parts[1:]) 
                    
                   
                    output_obj = {
                        "prompt": [
                            {"role": "user", "content": prompt_content},
                            {"role": "assistant", "content": label_content}
                        ]
                    }
                    f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                else:
                    tqdm.write(f"跳过：未找到分割符 {SPLIT_MARKER}")
                
            except Exception as e:
                tqdm.write(f"处理失败: {str(e)}")

    print(f"\n✅ 转换完成！数据已保存至: {output_path}")

if __name__ == "__main__":
    main()