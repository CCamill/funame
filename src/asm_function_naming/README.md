# ASM Function Naming Project

## 项目概述

本项目旨在给定汇编语言函数，自动生成其函数名。采用改进的架构设计，解决了原方案中的几个关键问题。

## 架构设计 Prefix Tuning + Projection + Adapter

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练阶段                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ASM Code ──► CLAP-ASM ──► Projection ──► Prefix Tokens        │
│                              Layer          (可学习)            │
│                                │                                │
│                                ▼                                │
│  Prompt: "Generate function name:" + Prefix ──► llama3.2  │
│                                                      │          │
│                                                      ▼          │
│                                              Function Name      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 两阶段训练

**阶段1：对比学习预对齐（可选）**
- 使用对比学习初步对齐CLAP-asm和Qwen的表示空间
- 帮助投影层更快收敛

**阶段2：端到端生成训练**
- 冻结Qwen主体，训练投影层和LoRA适配器
- 使用交叉熵损失训练函数名生成

## 项目结构

```
asm_function_naming/
├── README.md                 # 本文件
├── config.py                 # 配置文件
├── data/
│   └── sample_data.csv       # 示例数据格式
├── models/
│   ├── __init__.py
│   ├── clap_asm.py          # CLAP-ASM模型封装
│   ├── projection.py        # 投影层
│   └── asm_naming_model.py  # 主模型
├── dataset.py               # 数据集类
├── train_alignment.py       # 阶段1：对比学习对齐
├── train_generation.py      # 阶段2：生成训练
├── inference.py             # 推理脚本
└── utils.py                 # 工具函数
```

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 12.0+ (RTX 4090)
- transformers
- peft (LoRA)
- bitsandbytes (4bit量化)

## 安装

```bash
pip install torch transformers peft bitsandbytes accelerate datasets
```

## 数据格式

CSV文件应包含以下列：
- `src_code`: 源代码函数
- `asm_code`: 对应的汇编代码
- `func_name`: 函数名（标签）

## 使用方法

### 1. 准备数据
将你的CSV文件放到 `data/` 目录下

### 2. 阶段1：对比学习对齐（可选但推荐）
```bash
python train_alignment.py --data_path data/your_data.csv
```

### 3. 阶段2：生成训练
```bash
python train_generation.py --data_path data/your_data.csv
```

### 4. 推理
```bash
python inference.py --asm_code "your assembly code here"
```

## 注意事项

1. CLAP-ASM模型需要从HuggingFace下载或使用本地路径
2. 首次运行会下载llama3.2:3b模型
3. 4bit量化显著减少显存占用，适合RTX 4090
