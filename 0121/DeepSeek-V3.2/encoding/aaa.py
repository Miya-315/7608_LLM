# 
import json
# import pytest
# import os
# from typing import List, Dict, Any
# from encoding_dsv32 import (
#     encode_arguments_to_dsml,
#     decode_dsml_to_arguments,
#     tools_from_openai_format,
#     tool_calls_from_openai_format,
#     tool_calls_to_openai_format,
#     find_last_user_index,
#     render_message,
#     drop_thinking_messages,
#     to_json
# )

# # 读取JSONL前三条数据
# def load_test_data() -> List[Dict[str, Any]]:
#     data_path = "/gfs/space/chatrl/users/djy/0121/sft_code/250305-250529_math_combined-41649_no_repeat_70_turn1_deepseekv3.2.jsonl"
#     assert os.path.exists(data_path), f"数据文件不存在: {data_path}"
#     test_data = []
#     with open(data_path, "r", encoding="utf-8") as f:
#         for i, line in enumerate(f):
#             if i >= 3:
#                 break
#             test_data.append(json.loads(line.strip()))
#     assert len(test_data) >= 3, "JSONL文件前三条数据读取失败"
#     return test_data

# # 全局测试数据
# TEST_DATA = load_test_data()

# class TestEncodingDSV32:
#     """encoding_dsv32.py 核心逻辑单元测试"""

#     def test_find_last_user_index(self):
#         """测试找最后一条用户消息索引"""
#         # 用测试数据第一条构造消息列表
#         for data in TEST_DATA:
#             messages = data.get("messages", [])
#             if not messages:
#                 continue
#             last_user_idx = find_last_user_index(messages)
#             # 验证逻辑：最后一个role为user/developer的索引
#             user_indices = [i for i, msg in enumerate(messages) if msg.get("role") in ["user", "developer"]]
#             if user_indices:
#                 assert last_user_idx == user_indices[-1], f"最后用户索引匹配失败，预期{user_indices[-1]}，实际{last_user_idx}"
#             else:
#                 assert last_user_idx == -1, "无用户消息时应返回-1"

#     def test_tools_from_openai_format(self):
#         """测试OpenAI工具格式转自定义格式"""
#         for data in TEST_DATA:
#             tools = data.get("tools", [])
#             if not tools:
#                 continue
#             converted_tools = tools_from_openai_format(tools)
#             # 验证转换后结构：提取每个tool的function字段
#             assert len(converted_tools) == len(tools)
#             for idx, tool in enumerate(tools):
#                 assert converted_tools[idx] == tool["function"], "工具格式转换失败"

#     def test_tool_calls_format_conversion(self):
#         """测试工具调用的OpenAI格式<->自定义格式互转"""
#         for data in TEST_DATA:
#             tool_calls = data.get("tool_calls", [])
#             if not tool_calls:
#                 continue
#             # OpenAI -> 自定义
#             custom_tool_calls = tool_calls_from_openai_format(tool_calls)
#             assert len(custom_tool_calls) == len(tool_calls)
#             for idx, call in enumerate(tool_calls):
#                 assert custom_tool_calls[idx]["name"] == call["function"]["name"]
#                 assert custom_tool_calls[idx]["arguments"] == call["function"]["arguments"]
            
#             # 自定义 -> OpenAI
#             openai_tool_calls = tool_calls_to_openai_format(custom_tool_calls)
#             assert len(openai_tool_calls) == len(custom_tool_calls)
#             for idx, call in enumerate(openai_tool_calls):
#                 assert call["type"] == "function"
#                 assert call["function"]["name"] == custom_tool_calls[idx]["name"]
#                 assert call["function"]["arguments"] == custom_tool_calls[idx]["arguments"]

#     def test_encode_arguments_to_dsml(self):
#         """测试工具参数编码为DSML格式"""
#         for data in TEST_DATA:
#             tool_calls = data.get("tool_calls", [])
#             if not tool_calls:
#                 continue
#             custom_calls = tool_calls_from_openai_format(tool_calls)
#             for call in custom_calls:
#                 # 编码参数为DSML
#                 dsml_args = encode_arguments_to_dsml(call)
#                 # 验证DSML格式结构（包含参数名、string标识、值）
#                 assert "<｜DSML｜parameter name=" in dsml_args
#                 assert "string=" in dsml_args
#                 # 解析参数原始值，验证编码后值的正确性
#                 raw_args = json.loads(call["arguments"])
#                 for k, v in raw_args.items():
#                     if isinstance(v, str):
#                         assert f'string="true"' in dsml_args
#                         assert str(v) in dsml_args
#                     else:
#                         assert f'string="false"' in dsml_args
#                         assert to_json(v) in dsml_args

#     def test_render_message(self):
#         """测试不同角色消息的渲染逻辑"""
#         for data in TEST_DATA:
#             messages = data.get("messages", [])
#             if not messages:
#                 continue
#             for idx, msg in enumerate(messages):
#                 role = msg.get("role")
#                 # 测试chat模式渲染
#                 try:
#                     rendered = render_message(idx, messages, thinking_mode="chat")
#                     # 验证渲染结果非空，且包含核心token（如用户/助手分隔符）
#                     assert len(rendered) > 0
#                     if role == "user":
#                         assert "<｜User｜>" in rendered
#                     elif role == "assistant":
#                         assert "<｜end▁of▁sentence｜>" in rendered
#                     elif role == "system":
#                         assert msg.get("content", "") in rendered
#                 except NotImplementedError as e:
#                     # 忽略未实现的role（如果有）
#                     if "Unknown role" in str(e):
#                         continue
#                 # 测试thinking模式渲染
#                 try:
#                     rendered_thinking = render_message(idx, messages, thinking_mode="thinking")
#                     assert len(rendered_thinking) > 0
#                     last_user_idx = find_last_user_index(messages)
#                     if role in ["user", "developer"] and idx == last_user_idx:
#                         assert "" in rendered_thinking  # thinking_start_token
#                 except NotImplementedError as e:
#                     if "Unknown role" in str(e):
#                         continue

#     def test_drop_thinking_messages(self):
#         """测试移除思考内容的消息处理逻辑"""
#         for data in TEST_DATA:
#             messages = data.get("messages", [])
#             if not messages:
#                 continue
#             # 处理前备份原始数据
#             original_messages = json.loads(json.dumps(messages))
#             # 移除思考内容
#             messages_wo_thinking = drop_thinking_messages(messages)
#             # 验证长度一致（仅移除字段，不移除消息）
#             assert len(messages_wo_thinking) == len(messages)
#             # 验证assistant消息的reasoning_content被移除
#             for idx, msg in enumerate(messages_wo_thinking):
#                 if msg.get("role") == "assistant":
#                     assert "reasoning_content" not in msg
#                 else:
#                     # 其他角色消息内容不变
#                     assert msg == original_messages[idx]

# if __name__ == "__main__":
#     pytest.main(["-v", __file__])


from transformers import PreTrainedTokenizerFast

tok = PreTrainedTokenizerFast(tokenizer_file="/gfs/space/chatrl/users/djy/0121/DeepSeek-V3.2/tokenizer.json",
                              **{"unk_token":"<unk>"})

specials = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<think>", "</think>", "｜DSML｜", "<｜User｜>", "<｜Assistant｜>"]
added = tok.add_tokens(specials, special_tokens=True)
if added:
    print("Added special tokens:", added)
# 验证 round-trip
s = "<｜begin▁of▁sentence｜><｜User｜>示例文本<｜Assistant｜>"
ids = tok.encode(s)
print("decode:", tok.decode(ids))