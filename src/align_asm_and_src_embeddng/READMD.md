# Assembly Function Naming with Deep Learning

## 项目概述

本项目旨在通过深度学习模型从汇编代码生成函数名。核心思想是：
1. 使用CLAP-asm模型编码汇编代码
2. 使用Qwen2.5-coder编码源代码
3. 通过对比学习对齐两个embedding空间
4. 使用对齐后的汇编embedding生成函数名

## 架构设计

```
┌─────────────────┐
│  Assembly Code  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   CLAP-asm      │─────▶│   Adapter    │
│   (Encoder)     │      │  Projection  │
└─────────────────┘      └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Qwen2.5-    │
┌─────────────────┐      │   coder      │      ┌─────────────┐
│  Source Code    │─────▶│  (Layer 16)  │─────▶│ Function    │
└─────────────────┘      └──────────────┘      │ Name        │
                                                └─────────────┘
        Stage 1: Contrastive Learning
        Stage 2: Generation with LoRA
```

## 关键改进

### 1. 维度对齐问题
- **问题**: CLAP-asm和Qwen的hidden dimension可能不同
- **解决**: 添加Projection Adapter (多层MLP + LayerNorm)

### 2. 训练策略
- **阶段1**: 对比学习对齐embedding（训练CLAP-asm + Adapter）
- **阶段2**: 生成训练（冻结CLAP-asm + Adapter，LoRA微调Qwen）

### 3. 注入方式
- **原方案**: 直接注入第17层
- **改进**: 使用prefix-based方法，将asm embedding作为特殊token prepend到输入序列
- **优势**: 更稳定，与transformer架构更兼容

### 4. 量化策略
- 使用4-bit NF4量化加载Qwen2.5-coder-3B
- 节省显存，4090 24GB可以流畅运行
- 配合LoRA实现高效微调

## 安装

```bash
# 克隆项目
cd asm_function_naming

# 安装依赖
pip install -r requirements.txt

# 如果使用conda
conda create -n asm_naming python=3.10
conda activate asm_naming
pip install -r requirements.txt
```

## 数据准备

数据格式 (JSON):
```json
[
  {
    "asm_code": "push rbp\nmov rbp, rsp\n...",
    "src_code": "int add(int a, int b) { return a + b; }",
    "function_name": "add"
  },
  ...
]
```

### 生成示例数据

```bash
python data_utils.py
```

这将在`./data/`目录下创建train.json, val.json, test.json示例数据。

## 训练

### Stage 1: 对比学习

```bash
python train_stage1.py
```

**目标**:
- 正样本余弦相似度 → 1.0
- 负样本余弦相似度 → -1.0
- 检索准确率 → 95%+

**可调参数** (在`config.py`中):
- `stage1_epochs`: 训练轮数 (默认10)
- `stage1_batch_size`: 批大小 (默认16)
- `stage1_learning_rate`: 学习率 (默认1e-4)
- `stage1_temperature`: 对比学习温度 (默认0.07)

### Stage 2: 函数名生成

```bash
python train_stage2.py --stage1_checkpoint ./checkpoints/stage1_best_model
```

**可调参数**:
- `stage2_epochs`: 训练轮数 (默认5)
- `stage2_batch_size`: 批大小 (默认8)
- `stage2_learning_rate`: 学习率 (默认2e-5)
- `lora_r`: LoRA秩 (默认16)
- `lora_alpha`: LoRA缩放 (默认32)

## 推理

### 单个预测

```bash
python inference.py \
    --checkpoint ./checkpoints/stage2_best_model \
    --input "push rbp\nmov rbp, rsp\n..." \
    --output prediction.json
```

### 批量预测

```bash
python inference.py \
    --checkpoint ./checkpoints/stage2_best_model \
    --input_file ./data/test.json \
    --output predictions.json \
    --num_beams 4
```

## 项目结构

```
asm_function_naming/
├── config.py              # 配置文件
├── model.py               # 模型架构
├── data_utils.py          # 数据处理
├── train_stage1.py        # Stage 1训练
├── train_stage2.py        # Stage 2训练
├── inference.py           # 推理脚本
├── requirements.txt       # 依赖
├── README.md             # 本文件
├── data/                 # 数据目录
│   ├── train.json
│   ├── val.json
│   └── test.json
├── checkpoints/          # 模型检查点
└── outputs/              # 输出目录
```

## 配置说明

### ModelConfig
- `qwen_model_name`: Qwen模型名称
- `qwen_target_layer`: 提取embedding的层数 (默认16)
- `qwen_inject_layer`: 注入embedding的层数 (默认17)
- `clap_asm_hidden_size`: CLAP-asm隐藏层大小
- `adapter_hidden_size`: Adapter输出大小 (Qwen的hidden_size)
- `lora_r`: LoRA秩 (越大越强但参数更多)

### TrainingConfig
- `gradient_accumulation_steps`: 梯度累积步数 (节省显存)
- `mixed_precision`: 混合精度训练 ("fp16" 或 "bf16")
- `max_grad_norm`: 梯度裁剪阈值

## 性能优化建议

### 1. 显存优化
- ✅ 已使用4-bit量化
- ✅ 已使用LoRA (仅训练1-2%参数)
- ✅ 已使用梯度累积
- 可选: 使用gradient checkpointing (牺牲速度换显存)

### 2. 训练效率
- 使用更大的batch size (如果显存允许)
- 使用fp16或bf16混合精度
- 多GPU训练: 使用`accelerate`或`DeepSpeed`

### 3. 模型改进
- **更好的注入方式**: 实现custom forward hook在第17层直接注入
- **多层注入**: 不只在第17层，可以在多个层注入信息
- **Cross-attention**: 使用cross-attention机制融合asm和text信息
- **预训练**: 先在大规模汇编-源码对上预训练

### 4. 数据增强
- 汇编代码的变换 (寄存器重命名、指令重排等)
- 回译 (source → asm → source)
- 对比度增强 (同一函数的不同编译版本)

## 已知问题与解决方案

### 问题1: CLAP-asm模型不可用
**解决**: 代码中包含placeholder实现，使用简单的Transformer encoder

### 问题2: 生成的函数名质量不高
**可能原因**:
1. Stage 1对齐不充分 → 增加训练轮数，调整temperature
2. 数据质量问题 → 使用真实的汇编-源码对
3. 模型容量不足 → 增加adapter大小，增加LoRA秩

### 问题3: 显存不足
**解决方案**:
1. 减小batch size
2. 增加gradient accumulation
3. 使用更激进的量化 (int8)
4. 减小max_length

## 评估指标

### Stage 1
- **余弦相似度**: 正样本应接近1.0，负样本应接近-1.0
- **检索准确率**: top-1检索准确率应>95%

### Stage 2
- **Exact Match**: 完全匹配的比例
- **BLEU Score**: 文本相似度
- **Edit Distance**: 编辑距离
- **Semantic Similarity**: 语义相似度 (如果有预训练的函数名embedding)

## 扩展方向

1. **多模态输入**: 结合CFG、调用图等信息
2. **迁移学习**: 在不同ISA之间迁移 (x86 → ARM)
3. **增量学习**: 支持新类型函数的快速适应
4. **可解释性**: 可视化模型关注的汇编指令
5. **强化学习**: 使用RL fine-tune以优化特定指标

## 引用

如果使用CLAP-asm，请引用相关论文。
如果使用Qwen2.5-coder，请引用Qwen相关论文。

## License

MIT License (根据实际情况修改)

## 贡献

欢迎提交Issue和Pull Request!

## 联系方式

如有问题，请联系: [your-email]