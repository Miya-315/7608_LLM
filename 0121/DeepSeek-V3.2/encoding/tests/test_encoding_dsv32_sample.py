import os
import sys
import json

import pytest
import copy

# Ensure the encoding module path is importable
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
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


def test_encode_messages_first_three_records():
    records = read_first_n_records(DATA_PATH, 3)
    assert len(records) >= 1

    for rec in records:
        messages = rec.get("messages", [])
        # basic sanity
        assert isinstance(messages, list)
        if not messages:
            continue

        # sanitize messages: some JSONL records use a different `tool_calls` format
        msgs = copy.deepcopy(messages)
        for m in msgs:
            tc = m.get("tool_calls")
            if isinstance(tc, list) and tc:
                # if elements use keys like 'name' and 'arguments', convert
                if isinstance(tc[0], dict) and "function" not in tc[0] and "name" in tc[0]:
                    m["tool_calls"] = [{"function": {"name": t.get("name"), "arguments": t.get("arguments")}} for t in tc]

        # encode in chat mode
        prompt_chat = ed32.encode_messages(msgs, thinking_mode="chat")
        # should include BOS token when no context
        assert prompt_chat.startswith(ed32.bos_token)

        # first user message content should appear in the encoded prompt
        first_user = None
        for m in messages:
            if m.get("role") in ["user", "developer"]:
                first_user = m
                break
        if first_user and first_user.get("content"):
            assert first_user.get("content") in prompt_chat

        # encode in thinking mode with drop_thinking=True: reasoning_content
        # should be removed for assistant messages that occur BEFORE the last user message
        prompt_thinking = ed32.encode_messages(msgs, thinking_mode="thinking", drop_thinking=True)

        last_user_idx = ed32.find_last_user_index(msgs)
        for idx, m in enumerate(msgs):
            if m.get("role") == "assistant" and m.get("reasoning_content"):
                reasoning = m.get("reasoning_content")
                if reasoning and idx < last_user_idx:
                    assert reasoning not in prompt_thinking

