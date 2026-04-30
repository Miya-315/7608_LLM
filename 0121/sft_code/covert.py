#!/usr/bin/env python3
# convert_prompt_to_messages.py
# Usage:
#   python3 convert_prompt_to_messages.py \
#       /gfs/space/chatrl/users/djy/0121/sft_code/math_with_tools_slime_format.jsonl \
#       /gfs/space/chatrl/users/djy/0121/sft_code/data.jsonl

import json
import re
import sys
from pathlib import Path

WRAP_TAGS = [
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
    "<｜User｜>", "<｜Assistant｜>"
]

def strip_wrap(s: str) -> str:
    if not isinstance(s, str):
        return s
    for t in WRAP_TAGS:
        s = s.replace(t, "")
    return s.strip()

# match assistant marker (several possible encodings)
ASSISTANT_RE = re.compile(r"(?:<｜Assistant｜>|<\|Assistant\||<\｜Assistant\｜>|<Assistant>|__ASSISTANT__)", flags=re.I)

def prompt_to_messages(p):
    # If already list/dict, try to reuse
    if isinstance(p, list):
        # assume list of {"role","content"}
        return p
    if isinstance(p, dict):
        # maybe a single message
        return [p]

    if not isinstance(p, str):
        return []

    s = p.strip()
    # split on first assistant marker
    m = ASSISTANT_RE.split(s, maxsplit=1)
    if len(m) == 1:
        # no explicit assistant marker -> put entire text as user
        user_text = strip_wrap(s)
        return [{"role": "user", "content": user_text}]
    user_part, assistant_part = m[0], m[1]
    user_text = strip_wrap(user_part)
    assistant_text = strip_wrap(assistant_part)
    messages = []
    if user_text:
        messages.append({"role": "user", "content": user_text})
    if assistant_text:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages

def convert_file(fin_path: Path, fout_path: Path):
    cnt_in = cnt_out = 0
    with fin_path.open("r", encoding="utf-8") as inf, fout_path.open("w", encoding="utf-8") as outf:
        for line in inf:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # skip malformed lines
                continue
            cnt_in += 1
            # if messages exist, keep
            if "messages" in rec and isinstance(rec["messages"], list):
                out = rec
            else:
                prompt = rec.get("prompt")
                if prompt is None:
                    # nothing to convert; keep record as-is
                    out = rec
                else:
                    msgs = prompt_to_messages(prompt)
                    out = dict(rec)  # shallow copy
                    out["messages"] = msgs
                    # optionally remove original prompt to avoid confusion
                    if "prompt" in out:
                        out.pop("prompt", None)
            outf.write(json.dumps(out, ensure_ascii=False) + "\n")
            cnt_out += 1
    print(f"Converted {cnt_in} records -> wrote {cnt_out} records to {fout_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 convert_prompt_to_messages.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    fin = Path(sys.argv[1])
    fout = Path(sys.argv[2])
    if not fin.exists():
        print("Input file not found:", fin)
        sys.exit(2)
    convert_file(fin, fout)