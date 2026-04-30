# 项目说明

本项目用于整理和验证一份面向数学工具调用场景的 SFT 数据，并配合 slime 框架启动监督微调训练。

## 目标

- 构建适合工具增强数学推理的 SFT 训练集。
- 基于 slime 的 SFT 流程，验证数据是否能够直接用于训练。
- 为后续扩展更多题型、更多工具调用样式预留数据接口。

## 目录说明

- `sft_code/math_with_tools_slime_format.jsonl`：当前使用的训练数据。
- `sft_code/`：与数据处理、转换、切分相关的脚本和辅助文件。
- `DeepSeek-V3.2/`：模型相关文件与配置，供后续推理或训练流程参考。

## 数据介绍

训练数据位于 `sft_code/math_with_tools_slime_format.jsonl`，采用 JSONL 格式，每一行是一条样本。

### 字段结构

当前样本主要包含一个 `prompt` 字段，内容是一个消息列表，消息格式接近 OpenAI Chat 风格：

- `role`: 角色，通常是 `user` 或 `assistant`。
- `content`: 对应消息内容。

示例特征：

- 用户输入是数学题目或选择题。
- 助手输出包含思路和最终答案。
- 部分样本使用了特殊标记，如 `<｜begin▁of▁sentence｜>` 和 `<｜end▁of▁sentence｜>`。

### 数据使用注意

启动脚本 `slime/scripts/run-glm4.7-flash-sft.sh` 中当前配置的是 `--input-key messages`，而这份数据文件中看到的是 `prompt` 字段。

如果你直接使用这份数据训练，通常需要二选一：

- 将启动脚本里的 `--input-key messages` 改成 `--input-key prompt`。
- 或者先把数据预处理成脚本期望的 `messages` 字段。

## 启动命令

在 slime 仓库根目录下启动训练脚本：

```bash
bash scripts/run-glm4.7-flash-sft.sh
```

如果你在本机目录结构中运行这份项目说明，对应脚本路径为：`/Users/dingjiayi/Downloads/日志备份/0128/slime/scripts/run-glm4.7-flash-sft.sh`。

### 启动前需要检查的配置

脚本里有几处硬编码路径，运行前通常需要按你的环境修改：

- `BASE_FOLDER`
- `--hf-checkpoint`
- `--ref-load`
- `--load`
- `--save`
- `--prompt-data`

另外，脚本默认假设：

- 机器具备 8 张 GPU。
- 已安装并可用 `ray`、`sglang`、`Megatron-LM`。
- 多机环境下可以通过 `ssh root@<worker_ip>` 拉起 worker。

## 训练脚本说明

`run-glm4.7-flash-sft.sh` 主要做了这些事情：

1. 清理残留进程，避免上一次任务影响当前启动。
2. 设置 NCCL、Ray、CUDA 相关环境变量。
3. 加载模型配置脚本 `scripts/models/glm4.7-30B-A3B.sh`。
4. 配置 checkpoint、数据、性能、优化器等参数。
5. 启动 Ray Head 和 Worker。
6. 通过 `ray job submit` 提交 `train_async.py` 训练任务。

## 建议的使用方式

如果你的目标是先把数据跑通，建议按下面顺序检查：

1. 确认 JSONL 文件每行都能被正常解析。
2. 确认数据字段和脚本的 `--input-key` 一致。
3. 确认模型路径和保存路径可写。
4. 确认 Ray 和多机通信环境正常。

## 后续扩展

如果后面要继续扩数据，可以统一保持以下规范：

- 每行一个样本。
- 保持角色字段一致，避免同一数据集里混用多种结构。
- 题目、推理、答案尽量分层清晰，便于后续做过滤、切分和统计。

如果你希望，我也可以继续把这份 README 扩成更完整的版本，补上“环境准备”“数据样例”“字段规范”“常见问题”几个小节。