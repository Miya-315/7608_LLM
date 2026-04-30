import os
import sys
import json
import copy
import difflib

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(HERE)
sys.path.insert(0, ROOT)

import encoding_dsv32 as ed32

DATA_PATH = "/gfs/space/chatrl/users/djy/0121/sft_code/250305-250529_math_combined-41649_no_repeat_70_turn1_deepseekv3.2.jsonl"


def read_first_n_records(path, n=3):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            records.append(json.loads(line))
    return records


def sanitize_messages(messages):
    msgs = copy.deepcopy(messages)
    for m in msgs:
        tc = m.get("tool_calls")
        if isinstance(tc, list) and tc:
            if isinstance(tc[0], dict) and "function" not in tc[0] and "name" in tc[0]:
                m["tool_calls"] = [{"function": {"name": t.get("name"), "arguments": t.get("arguments")}} for t in tc]
    return msgs


def concat_message_contents(messages):
    parts = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        reasoning = m.get("reasoning_content") or ""
        parts.append(f"[{role}] {content}")
        if reasoning:
            parts.append(f"[{role}][reasoning] {reasoning}")
    return "\n".join(parts)


def show_samples(n=3):
    records = read_first_n_records(DATA_PATH, n)
    for i, rec in enumerate(records, 1):
        print("="*60)
        print(f"Record {i}")
        messages = rec.get("messages", [])
        print(f"Original messages: {len(messages)} entries")
        print()

        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if content:
                print(f"- role={role}: {repr(content)[:300]}")
            if m.get("reasoning_content"):
                print(f"  reasoning: {repr(m.get('reasoning_content'))[:300]}")

        msgs = sanitize_messages(messages)

        prompt_chat = ed32.encode_messages(msgs, thinking_mode="chat")
        prompt_thinking = ed32.encode_messages(msgs, thinking_mode="thinking", drop_thinking=True)

        print()
        print(f"Encoded (chat) length: {len(prompt_chat)}")
        print(repr(prompt_chat)[:1000])
        print()
        print(f"Encoded (thinking, drop): length: {len(prompt_thinking)}")
        print(repr(prompt_thinking)[:1000])

        # show a short unified diff between concatenated original contents and encoded chat prompt
        orig_text = concat_message_contents(messages).splitlines()
        enc_lines = prompt_chat.splitlines()
        diff = list(difflib.unified_diff(orig_text, enc_lines, lineterm=""))
        print()
        print("Diff (original -> encoded) first 200 lines:")
        for line in diff[:200]:
            print(line)


if __name__ == "__main__":
    show_samples(3)
