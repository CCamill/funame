"""
配置文件 - ASM Function Naming Project
"""
from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class ModelConfig:
    """模型配置"""
    # CLAP-ASM配置
    # 注意：如果没有预训练的CLAP-ASM，我们使用CodeBERT作为替代
    # 你可以将这里替换为实际的CLAP-ASM模型路径
    clap_asm_model_name: str = "hustcw/clap-asm"  # 或你的CLAP-ASM模型路径
    clap_asm_hidden_size: int = 768  # CLAP-ASM的隐藏层维度
    
    # Qwen配置
    src_model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    src_hidden_size: int = 2048  # Qwen2.5-Coder-3B的隐藏层维度
    
    # 4bit量化配置
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True
    projection_dim: int = 768
    
    # 投影层配置
    num_prefix_tokens: int = 32  # prefix token数量
    projection_hidden_size: int = 1024  # 投影层中间维度
    projection_dropout: float = 0.1

    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据配置
    train_data_path: str = "data/sample_data.csv"
    eval_data_path: str = "data/sample_data.csv"
    max_asm_length: int = 768  # 汇编代码最大长度
    max_src_length: int = 1536  # 源代码最大长度（Qwen模型最大长度）
    max_name_length: int = 64   # 函数名最大长度
    
    # 阶段1：对比学习配置
    alignment_epochs: int = 10
    alignment_train_batch_size: int = 8  # 减小batch size以避免OOM
    alignment_eval_batch_size: int = 8
    alignment_gradient_accumulation_steps: int = 1  # 梯度累积，有效batch size = 8 * 4 = 32
    alignment_lr: float = 1e-4
    alignment_temperature: float = 0.07  # InfoNCE温度
    alignment_warmup_ratio: float = 0.1
    
    # 阶段2：生成训练配置
    generation_epochs: int = 20
    generation_train_batch_size: int = 4  # 小batch size因为使用4090
    generation_eval_batch_size: int = 4  # 小batch size因为使用4090
    generation_lr: float = 2e-5
    generation_warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 8
    
    # 通用配置
    seed: int = 42
    output_dir: str = "outputs"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    num_workers: int = 1
    

@dataclass
class InferenceConfig:
    """推理配置"""
    checkpoint_path: str = "outputs/best_model"
    max_new_tokens: int = 64
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False  # 函数名生成通常用贪婪解码
    num_beams: int = 3


def get_model_config() -> ModelConfig:
    return ModelConfig()


def get_training_config() -> TrainingConfig:
    return TrainingConfig()


def get_inference_config() -> InferenceConfig:
    return InferenceConfig()
