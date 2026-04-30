#!/usr/bin/env python3
import argparse
import json
import os
import copy
from tqdm import tqdm

try:
    from encoding_dsv32 import encode_messages
except ImportError:
    raise


def fix_data_structure(messages, tools):
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


def convert_file(input_path: str, output_path: str, thinking_mode: str = "thinking"):
    SPLIT_MARKER = "<｜Assistant｜>"

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        raise

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, total=total_lines, desc="Converting to DeepSeek chat SFT"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            messages = data.get("messages", [])
            tools = data.get("tools", [])

            fixed_messages = fix_data_structure(messages, tools)

            encode_config = dict(thinking_mode=thinking_mode, drop_thinking=False, add_default_bos_token=True)
            try:
                full_text = encode_messages(fixed_messages, **encode_config)
            except Exception:
                # fallback: join user/assistant content
                full_text = ""
                for m in fixed_messages:
                    role = m.get("role")
                    content = m.get("content", "")
                    if role and content:
                        full_text += content

            if SPLIT_MARKER in full_text:
                parts = full_text.split(SPLIT_MARKER)
                prompt_content = parts[0] + SPLIT_MARKER
                label_content = SPLIT_MARKER.join(parts[1:])

                output_obj = {
                    "messages": [
                        {"role": "user", "content": prompt_content},
                        {"role": "assistant", "content": label_content}
                    ]
                }
                f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
            else:
                # try to build messages directly from fixed_messages
                out_msgs = []
                for m in fixed_messages:
                    role = m.get("role")
                    content = m.get("content")
                    if role and content:
                        out_msgs.append({"role": role, "content": content})

                if out_msgs:
                    f_out.write(json.dumps({"messages": out_msgs}, ensure_ascii=False) + '\n')


def cli():
    parser = argparse.ArgumentParser(description="Convert raw jsonl to DeepSeek v3.2 chat SFT format")
    parser.add_argument("--input", "-i", required=False, default="/gfs/space/chatrl/users/hxh/code/agent_scripts/data/output/250305-250529_math_combined-41649_no_repeat_70_turn1_deepseekv3.2.jsonl", help="输入原始 jsonl 路径")
    parser.add_argument("--output", "-o", required=False, default="/gfs/space/chatrl/users/djy/0121/DeepSeek-V3.2/encoding/converted_for_sft.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--thinking", required=False, default="thinking", help="thinking_mode: 'thinking' or 'chat'")

    args = parser.parse_args()
    convert_file(args.input, args.output, thinking_mode=args.thinking)


if __name__ == '__main__':
    cli()
