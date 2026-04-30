import json
import os
from tqdm import tqdm
import copy
# 导入原始编码函数
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
                    # 假设原始结构是 {"name": "...", "arguments": "..."}
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
    input_path = "/gfs/space/chatrl/users/hxh/code/agent_scripts/data/output/250305-250529_math_combined-41649_no_repeat_70_turn1_deepseekv3.2.jsonl"
    output_path = "/gfs/space/chatrl/users/djy/0121/sft_code/math_with_tools_sft_data.jsonl"

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # 统计行数
    print("正在扫描文件...")
    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="全量转换中"):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                
                # 核心：修复数据结构
                fixed_messages = fix_data_structure(
                    data.get("messages", []), 
                    data.get("tools", [])
                )

                # 编码配置
                encode_config = dict(
                    thinking_mode="thinking", 
                    drop_thinking=False, 
                    add_default_bos_token=True
                )

                # 调用原始 encoding 函数
                formatted_prompt = encode_messages(fixed_messages, **encode_config)
                
                # 输出结果
                f_out.write(json.dumps({"text": formatted_prompt}, ensure_ascii=False) + '\n')
                
            except Exception as e:
                # 打印更详细的错误信息
                tqdm.write(f"处理失败跳过一条: 错误类型={type(e).__name__}, 内容={str(e)}")

    print(f"\n✅ 转换完成！")
    print(f"原始文件: {input_path}")
    print(f"目标文件: {output_path}")

if __name__ == "__main__":
    main()