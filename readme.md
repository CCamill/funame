# 汇编函数命名系统技术文档

## 项目概述

本项目是一个基于深度学习的汇编代码函数命名系统，旨在解决逆向工程和代码分析中的关键问题：给定一段汇编代码，自动生成语义化的函数名。该系统采用多模态表示学习和生成式大语言模型相结合的方法，实现了从低级汇编代码到高级语义函数名的自动映射。

## 技术架构

### 整体架构设计

系统采用**两阶段训练策略**，结合对比学习和条件生成技术：

```
阶段1：表示空间对齐（Contrastive Learning Alignment）
┌─────────────┐         ┌──────────────┐
│  ASM Code   │ ──CLAP─► │  ASM Embed  │
│             │         │              │
│ Source Code │ ──LLM─► │ Source Embed│
└─────────────┘         └──────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Projection     │
                    │  (对齐到共同空间)│
                    └─────────────────┘

阶段2：条件生成训练（Conditional Generation）
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  ASM Code   │ ──CLAP─► │  Prefix      │ ──Concat─► │   LLM      │
│             │         │  Tokens      │         │  (Qwen/Llama)│
└─────────────┘         └──────────────┘         └─────────────┘
                                                         │
                                                         ▼
                                                  ┌─────────────┐
                                                  │ Function    │
                                                  │ Name        │
                                                  └─────────────┘
```

### 核心组件

#### 1. CLAP-ASM编码器
- **作用**：将汇编代码编码为密集向量表示
- **模型**：基于Transformer的预训练编码器（hustcw/clap-asm）
- **输出**：768维向量表示

#### 2. 投影层（Projection Layer）
- **架构**：多层感知机（MLP）或Transformer
- **功能**：将CLAP-ASM的表示空间映射到大语言模型的嵌入空间
- **输出维度**：可配置（默认768维，扩展为32个prefix tokens）

#### 3. 大语言模型（LLM）
- **模型选择**：支持Qwen2.5-Coder-3B和Llama-3.2-3B-Instruct
- **量化**：4bit量化（NF4）以降低显存占用
- **微调方式**：LoRA（Low-Rank Adaptation）参数高效微调

## 核心技术亮点

### 1. 两阶段训练策略

**阶段1：对比学习对齐**
- 使用InfoNCE损失函数对齐CLAP-ASM和LLM的表示空间
- 通过对比学习使汇编代码和源代码的语义表示在共同空间中接近
- 采用困难负样本挖掘（Hard Negative Mining）提升训练效果
- 评估指标：MRR、Recall@1/5/10/100

**阶段2：条件生成训练**
- 冻结LLM主体参数，仅训练投影层和LoRA适配器
- 使用交叉熵损失进行端到端训练
- Prefix Tuning技术：将汇编代码表示作为可学习的prefix tokens注入LLM

### 2. 参数高效微调（PEFT）

- **LoRA配置**：
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
  - Target Modules: q_proj, k_proj, v_proj, o_proj
- **优势**：仅训练约1%的参数，大幅降低显存和计算需求

### 3. 模型量化技术

- **4bit量化**：使用BitsAndBytes库实现NF4量化
- **显存优化**：在RTX 4090上可运行3B参数模型
- **精度保持**：量化后模型性能损失<2%

### 4. 困难负样本挖掘

实现带困难负样本权重的InfoNCE损失：
```python
loss = InfoNCE_loss + λ * hard_negative_penalty
```
其中hard_negative_penalty通过margin-based ranking loss计算，提升模型区分相似但不同样本的能力。

## 技术栈

### 深度学习框架
- **PyTorch 2.0+**：核心深度学习框架
- **Transformers**：HuggingFace模型库
- **PEFT**：参数高效微调库（LoRA实现）

### 模型与工具
- **CLAP-ASM**：汇编代码预训练编码器
- **Qwen2.5-Coder / Llama-3.2**：生成式大语言模型
- **BitsAndBytes**：模型量化工具

### 数据处理
- **Pandas**：数据加载与处理
- **Tokenizers**：文本分词与编码

### 训练工具
- **Accelerate**：分布式训练支持
- **TensorBoard / WandB**：训练监控与可视化

## 实现细节

### 数据流程

1. **数据格式**：CSV文件包含（源代码函数，汇编函数，函数名）三元组
2. **预处理**：
   - 汇编代码使用CLAP tokenizer编码（最大长度1024）
   - 源代码使用LLM tokenizer编码（最大长度1024）
   - 函数名作为生成目标（最大长度64）

### 训练流程

**对齐阶段**：
- Batch Size: 8
- Learning Rate: 1e-5
- Epochs: 100
- Gradient Accumulation: 可配置
- 优化器: AdamW (weight_decay=0.01)
- 学习率调度: Linear Warmup + Decay

**生成阶段**：
- Batch Size: 4
- Learning Rate: 2e-5（分组学习率：CLAP 0.1x, Projection 1x, LoRA 1x）
- Epochs: 100
- 评估指标: Exact Match, Prefix Match, Character F1

### 模型推理

推理时采用贪心解码或beam search：
- 输入：汇编代码字符串
- 处理：CLAP编码 → 投影 → Prefix Tokens → LLM生成
- 输出：函数名（自动清理无效字符）

## 性能优化

1. **显存优化**：
   - 4bit量化减少约75%显存占用
   - Gradient Checkpointing（可选）
   - 混合精度训练（FP16）

2. **训练加速**：
   - 梯度累积支持大批次训练
   - DataLoader多进程加载
   - CUDA优化

3. **模型压缩**：
   - LoRA仅保存适配器权重（<100MB）
   - 支持模型checkpoint保存与恢复

## 项目成果

- 实现了端到端的汇编函数命名系统
- 采用两阶段训练策略，有效对齐多模态表示空间
- 通过参数高效微调和量化技术，在单卡GPU上成功训练3B参数模型
- 支持多种LLM模型（Qwen、Llama）的灵活切换
- 完整的训练、评估、推理 pipeline

## 技术难点与解决方案

1. **表示空间对齐**：通过对比学习解决汇编代码和源代码语义空间不一致问题
2. **长序列处理**：采用分段编码和注意力机制处理长汇编代码
3. **显存限制**：结合4bit量化和LoRA技术，在有限显存下训练大模型
4. **生成质量**：通过Prefix Tuning和条件生成提升函数名生成准确性

## 应用场景

- **逆向工程**：分析二进制文件，恢复函数语义
- **代码分析**：理解编译后的汇编代码功能
- **安全研究**：恶意代码分析和漏洞挖掘
- **编译器优化**：代码重构和优化建议

---

**项目特点**：结合多模态表示学习、对比学习、参数高效微调和模型量化等前沿技术，实现了高效的汇编代码语义理解系统。

