"""
Configuration file for Assembly Function Naming Project
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration"""
    # Qwen2.5-coder configuration
    qwen_model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    qwen_target_layer: int = 16  # Extract embedding from this layer
    qwen_inject_layer: int = 17  # Inject asm embedding to this layer
    qwen_quantization: str = "4bit"  # Use 4-bit quantization for 4090
    
    # CLAP-asm configuration
    clap_asm_model_name: str = "CLAP-asm"  # Replace with actual model name/path
    clap_asm_hidden_size: int = 768  # Adjust based on actual model
    
    # Adapter configuration
    adapter_hidden_size: int = 2048  # Qwen2.5-coder-3b hidden size
    adapter_dropout: float = 0.1
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Will be set in code
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Stage 1: Contrastive learning
    stage1_epochs: int = 10
    stage1_batch_size: int = 16
    stage1_learning_rate: float = 1e-4
    stage1_temperature: float = 0.07
    stage1_margin: float = 0.5
    
    # Stage 2: Function name generation
    stage2_epochs: int = 5
    stage2_batch_size: int = 8
    stage2_learning_rate: float = 2e-5
    stage2_max_length: int = 32  # Max function name length
    
    # General
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    # Device
    device: str = "cuda"
    mixed_precision: str = "fp16"
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


@dataclass
class DataConfig:
    """Data configuration"""
    train_data_path: str = "./data/train.json"
    val_data_path: str = "./data/val.json"
    test_data_path: str = "./data/test.json"
    
    # Data format: {"asm_code": "...", "src_code": "...", "function_name": "..."}
    max_asm_length: int = 2048
    max_src_length: int = 1024
    
    # Augmentation
    use_data_augmentation: bool = False
    num_negatives: int = 4  # Number of negative samples for contrastive learning