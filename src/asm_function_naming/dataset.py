"""
数据集类

处理(源码函数, 汇编函数, 函数名)三元组
"""
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import PreTrainedTokenizer
from typing import Optional, Dict, List, Tuple, Any
import random


class AsmFunctionDataset(Dataset):
    """
    ASM函数命名数据集
    
    用于生成训练阶段
    """
    
    def __init__(
        self,
        data_path: str,
        clap_tokenizer: PreTrainedTokenizer,
        qwen_tokenizer: PreTrainedTokenizer,
        max_asm_length: int = 2048,
        max_prompt_length: int = 256,
        max_name_length: int = 64,
        prompt_template: str = "Based on the assembly function semantics, the function name is:"
    ):
        """
        Args:
            data_path: CSV文件路径
            clap_tokenizer: CLAP-ASM的tokenizer
            qwen_tokenizer: Qwen的tokenizer
            max_asm_length: ASM代码最大长度
            max_prompt_length: 提示最大长度
            max_name_length: 函数名最大长度
            prompt_template: 提示模板
        """
        self.clap_tokenizer = clap_tokenizer
        self.qwen_tokenizer = qwen_tokenizer
        self.max_asm_length = max_asm_length
        self.max_prompt_length = max_prompt_length
        self.max_name_length = max_name_length
        self.prompt_template = prompt_template
        
        # 加载数据
        self.df = pd.read_csv(data_path)
        
        # 确保必要的列存在
        required_columns = ["src_func", "asm_func", "func_name"]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"CSV文件缺少必要的列: {col}")
                
        print(f"Loaded {len(self.df)} samples from {data_path}")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        asm_func = str(row["asm_func"])
        func_name = str(row["func_name"])
        
        # 1. 编码ASM代码（用于CLAP）
        asm_encoding = self.clap_tokenizer(
            asm_func,
            max_length=self.max_asm_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 2. 构建prompt和target
        # 格式: "[PROMPT][FUNCTION_NAME]<eos>"
        full_text = f"{self.prompt_template} {func_name}"
        
        encoding = self.qwen_tokenizer(
            full_text,
            max_length=self.max_prompt_length + self.max_name_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 3. 创建labels（只在函数名部分计算loss）
        # 首先找到函数名开始的位置
        prompt_encoding = self.qwen_tokenizer(
            self.prompt_template,
            return_tensors="pt"
        )
        prompt_length = prompt_encoding["input_ids"].size(1)
        
        labels = encoding["input_ids"].clone()
        # 将prompt部分的label设为-100
        labels[0, :prompt_length] = -100
        # 将padding部分的label设为-100
        labels[labels == self.qwen_tokenizer.pad_token_id] = -100
        
        return {
            "asm_input_ids": asm_encoding["input_ids"].squeeze(0),
            "asm_attention_mask": asm_encoding["attention_mask"].squeeze(0),
            "prompt_input_ids": encoding["input_ids"].squeeze(0),
            "prompt_attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "func_name": func_name
        }


class  AsmAlignmentDataset(Dataset):
    """
    对比学习对齐数据集
    
    用于对齐CLAP-ASM和Qwen的表示空间
    """
    
    def __init__(
        self,
        data_path: str,
        clap_tokenizer: PreTrainedTokenizer,
        src_tokenizer: PreTrainedTokenizer,
        max_asm_length: int = 768,
        max_src_length: int = 1536
    ):
        self.clap_tokenizer = clap_tokenizer
        self.src_tokenizer = src_tokenizer

        self.asm_prompt_template = "Analyze the following assembly code and understand its semantic meaning:\n"\
            "```asm\n{code}\n```\n"\
            "The keyword for the semantics of this function is:"

        self.max_asm_length = max_asm_length
        self.max_src_length = max_src_length
        
        self.df = pd.read_csv(data_path)
        # self.df = self.df[:100]

        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        asm_func = str(row["asm_func"])
        src_func = str(row["src_func"])
        
        # 编码ASM
        # CLAP tokenizer 不支持 truncation 参数，需要先编码再手动截断和填充
        asm_encoding = self.clap_tokenizer(
            [{'function': asm_func}],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_asm_length
        )
        
        # 手动截断和填充
        asm_input_ids = asm_encoding["input_ids"].squeeze(0)  # [seq_len]
        asm_attention_mask = asm_encoding["attention_mask"].squeeze(0)  # [seq_len]
        
        # 截断到最大长度
        if asm_input_ids.size(0) > self.max_asm_length:
            asm_input_ids = asm_input_ids[:self.max_asm_length]
            asm_attention_mask = asm_attention_mask[:self.max_asm_length]
        
        # 填充到最大长度
        pad_length = self.max_asm_length - asm_input_ids.size(0)
        if pad_length > 0:
            pad_token_id = self.clap_tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.clap_tokenizer.eos_token_id if self.clap_tokenizer.eos_token_id is not None else 0
            
            asm_input_ids = torch.cat([
                asm_input_ids, 
                torch.full((pad_length,), pad_token_id, dtype=asm_input_ids.dtype)
            ])
            asm_attention_mask = torch.cat([
                asm_attention_mask, 
                torch.zeros(pad_length, dtype=asm_attention_mask.dtype)
            ])
        
        # 确保长度为 max_asm_length
        assert asm_input_ids.size(0) == self.max_asm_length, f"ASM input_ids length mismatch: {asm_input_ids.size(0)} != {self.max_asm_length}"
        assert asm_attention_mask.size(0) == self.max_asm_length, f"ASM attention_mask length mismatch: {asm_attention_mask.size(0)} != {self.max_asm_length}"
        
        # 编码源代码
        # max_src_length 已经在 __init__ 中确保不超过模型最大长度
        src_encoding = self.src_tokenizer(
            self.asm_prompt_template.format(code=src_func),
            max_length=self.max_src_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "asm_input_ids": asm_input_ids,
            "asm_attention_mask": asm_attention_mask,
            "src_input_ids": src_encoding["input_ids"].squeeze(0),
            "src_attention_mask": src_encoding["attention_mask"].squeeze(0)
        }


class HardNegativeDataset(Dataset):
    """
    带有难负样本的对比学习数据集
    
    使用在batch内随机打乱的负样本
    """
    
    def __init__(
        self,
        data_path: str,
        clap_tokenizer: PreTrainedTokenizer,
        qwen_tokenizer: PreTrainedTokenizer,
        max_asm_length: int = 2048,
        max_src_length: int = 1024,
        num_negatives: int = 4
    ):
        self.base_dataset = AsmAlignmentDataset(
            data_path=data_path,
            clap_tokenizer=clap_tokenizer,
            src_tokenizer=qwen_tokenizer,
            max_asm_length=max_asm_length,
            max_src_length=max_src_length
        )
        self.num_negatives = num_negatives
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 获取正样本
        positive = self.base_dataset[idx]
        
        # 随机选择负样本索引
        all_indices = list(range(len(self.base_dataset)))
        all_indices.remove(idx)
        neg_indices = random.sample(all_indices, min(self.num_negatives, len(all_indices)))
        
        # 获取负样本的源代码
        neg_src_input_ids = []
        neg_src_attention_mask = []
        
        for neg_idx in neg_indices:
            neg_sample = self.base_dataset[neg_idx]
            neg_src_input_ids.append(neg_sample["src_input_ids"])
            neg_src_attention_mask.append(neg_sample["src_attention_mask"])
            
        return {
            **positive,
            "neg_src_input_ids": torch.stack(neg_src_input_ids) if neg_src_input_ids else None,
            "neg_src_attention_mask": torch.stack(neg_src_attention_mask) if neg_src_attention_mask else None
        }


def create_dataloaders(
    train_data_path: str,
    eval_data_path: str,
    clap_tokenizer: PreTrainedTokenizer,
    src_tokenizer: PreTrainedTokenizer,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    dataset_type: str = "generation",  # "generation" or "alignment"
    num_workers: int = 4,
    logger: logging.Logger = None,
    max_asm_length: int = 768,
    max_src_length: int = 1536,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证DataLoader
    
    Args:
        data_path: 数据路径
        clap_tokenizer: CLAP tokenizer
        qwen_tokenizer: Qwen tokenizer
        train_batch_size: train batch大小
        eval_batch_size: eval batch大小
        train_split: 训练集比例
        dataset_type: 数据集类型
        num_workers: 工作进程数
        **dataset_kwargs: 传递给数据集的其他参数
        
    Returns:
        train_loader, val_loader
    """
    
    # 创建数据集
    if dataset_type == "generation":
        DatasetClass = AsmFunctionDataset
    elif dataset_type == "alignment":
        DatasetClass = AsmAlignmentDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
        
    train_dataset = DatasetClass(
        data_path=train_data_path,
        clap_tokenizer=clap_tokenizer,
        src_tokenizer=src_tokenizer,
        max_asm_length=max_asm_length,
        max_src_length=max_src_length,
        **dataset_kwargs
    )
    logger.info(f"training dataset: {len(train_dataset)} samples")
    val_dataset = DatasetClass(
        data_path=eval_data_path,
        clap_tokenizer=clap_tokenizer,
        src_tokenizer=src_tokenizer,
        max_asm_length=max_asm_length,
        max_src_length=max_src_length,
        **dataset_kwargs
    )
    logger.info(f"evaluation dataset: {len(val_dataset)} samples")
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

