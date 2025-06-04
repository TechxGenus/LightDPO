# LightDPO

轻量级 Direct Preference Optimization (DPO) 训练脚本集合，包含数据准备和训练流程。

## 目录说明
- `prepare_conversation.py`：将原始对话整理为模型输入格式。
- `prepare_tokenize.py`：使用给定 tokenizer 对整理后的数据进行分词。
- `precompute_reference_logprobs.py`：用参考模型计算序列概率，可用于 reference-free 训练。
- `training.py`：主训练脚本，使用 DPO 方法微调语言模型。
- `fused_dpo_loss.py`：利用 `torch.compile` 加速的 DPO 损失实现。

## 使用示例
1. 准备对话数据并进行分词：
   ```bash
   python prepare_conversation.py --model_name_or_path <模型名> --save_path data.json
   python prepare_tokenize.py --model_name_or_path <模型名> --input_path data.json --output_path tokenized.json
   ```
2. （可选）预计算参考模型概率：
   ```bash
   python precompute_reference_logprobs.py --model_name_or_path <参考模型名> \
       --input_path tokenized.json --output_path with_ref.json
   ```
3. 开始训练：
   ```bash
   python training.py \
       --model_name_or_path <模型名> \
       --data_path tokenized.json \
       --output_dir output
   ```

以上命令仅为基本示例，具体参数请根据需要调整。

