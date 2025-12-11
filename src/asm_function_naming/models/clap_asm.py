"""
CLAP-ASM模型封装

由于原始CLAP-ASM模型可能不在HuggingFace上公开，
我们提供两种选择：
1. 使用CodeBERT作为替代（默认）
2. 使用你自己的CLAP-ASM模型权重

如果你有CLAP-ASM的预训练权重，可以替换load_pretrained方法
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict, Any


class CLAPAsmEncoder(nn.Module):
    """
    汇编语言编码器
    
    支持两种模式：
    1. 使用预训练的CodeBERT（适合没有CLAP-ASM的情况）
    2. 加载自定义的CLAP-ASM权重
    """
    
    def __init__(
        self,
        model_name_or_path: str = "hustcw/clap-asm",
        hidden_size: int = 768,
        pooling_strategy: str = "mean",  # "cls", "mean", "max"
        freeze: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pooling_strategy = pooling_strategy
        
        # 加载模型和tokenizer
        self.encoder = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # 确保有pad token   
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 是否冻结参数
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 获取last_hidden_state
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # 池化策略
        if self.pooling_strategy == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.pooling_strategy == "mean":
            # 考虑attention mask的平均池化
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif self.pooling_strategy == "max":
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states[~mask_expanded.bool()] = float('-inf')
            pooled = torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
        return pooled
    
    def encode(
        self,
        asm_code: str,
        max_length: int = 2048,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        便捷方法：直接从汇编代码字符串获取embedding
        
        Args:
            asm_code: 汇编代码字符串
            max_length: 最大长度
            device: 设备
            
        Returns:
            embedding: [1, hidden_size]
        """
        if device is None:
            device = next(self.parameters()).device
            
        inputs = self.tokenizer(
            asm_code,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embedding = self.forward(**inputs)
            
        return embedding
    
    def encode_batch(
        self,
        asm_codes: list,
        max_length: int = 2048,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        批量编码汇编代码
        
        Args:
            asm_codes: 汇编代码字符串列表
            max_length: 最大长度
            device: 设备
            
        Returns:
            embeddings: [batch_size, hidden_size]
        """
        if device is None:
            device = next(self.parameters()).device
            
        inputs = self.tokenizer(
            asm_codes,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return self.forward(**inputs)
    
    @classmethod
    def load_clap_asm(
        cls,
        checkpoint_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> "CLAPAsmEncoder":
        """
        加载自定义CLAP-ASM权重
        
        如果你有预训练的CLAP-ASM模型，使用此方法加载
        
        Args:
            checkpoint_path: 权重文件路径
            config: 模型配置
            
        Returns:
            CLAPAsmEncoder实例
        """
        # 这里需要根据你的CLAP-ASM模型结构进行适配
        # 以下是一个示例实现
        
        if config is None:
            config = {}
            
        model = cls(
            model_name_or_path=config.get("base_model", "microsoft/codebert-base"),
            hidden_size=config.get("hidden_size", 768),
            pooling_strategy=config.get("pooling_strategy", "cls")
        )
        
        # 加载权重
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.encoder.load_state_dict(state_dict, strict=False)
        
        return model


class CLAPAsmEncoderWithProjection(nn.Module):
    """
    带投影头的CLAP-ASM编码器
    用于对比学习阶段
    """
    
    def __init__(
        self,
        encoder: CLAPAsmEncoder,
        projection_dim: int = 512
    ):
        super().__init__()
        self.encoder = encoder
        
        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(encoder.hidden_size, encoder.hidden_size),
            nn.GELU(),
            nn.Linear(encoder.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> tuple:
        """
        Returns:
            embedding: 原始embedding
            projected: 投影后的embedding（用于对比学习）
        """
        embedding = self.encoder(input_ids, attention_mask, **kwargs)
        projected = self.projection(embedding)
        projected = nn.functional.normalize(projected, p=2, dim=-1)
        
        return embedding, projected
