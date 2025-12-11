"""
投影层模块

将CLAP-ASM的embedding投影到Qwen的输入空间
支持多种投影策略
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class AsmProjection(nn.Module):
    """
    汇编embedding到LLM prefix tokens的投影层
    
    将CLAP-ASM的单个embedding [batch, clap_hidden]
    转换为多个prefix tokens [batch, num_prefix, llm_hidden]
    """
    
    def __init__(
        self,
        clap_hidden_size: int = 768,
        llm_hidden_size: int = 2048,
        num_prefix_tokens: int = 32,
        projection_hidden_size: int = 1024,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.clap_hidden_size = clap_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.num_prefix_tokens = num_prefix_tokens
        
        # 多层投影网络
        self.projection = nn.Sequential(
            nn.Linear(clap_hidden_size, projection_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_size, projection_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_size, num_prefix_tokens * llm_hidden_size)
        )
        
        # 可选的层归一化
        self.layer_norm = nn.LayerNorm(llm_hidden_size) if use_layer_norm else nn.Identity()
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, asm_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            asm_embedding: [batch_size, clap_hidden_size]
            
        Returns:
            prefix_tokens: [batch_size, num_prefix_tokens, llm_hidden_size]
        """
        batch_size = asm_embedding.size(0)
        
        # 投影
        projected = self.projection(asm_embedding)
        
        # 重塑为prefix tokens
        prefix_tokens = projected.view(batch_size, self.num_prefix_tokens, self.llm_hidden_size)
        
        # 层归一化
        prefix_tokens = self.layer_norm(prefix_tokens)
        
        return prefix_tokens


class CrossAttentionProjection(nn.Module):
    """
    基于交叉注意力的投影层（可选的高级投影方式）
    
    使用可学习的查询向量与ASM embedding做交叉注意力
    生成更具表达力的prefix tokens
    """
    
    def __init__(
        self,
        clap_hidden_size: int = 768,
        llm_hidden_size: int = 2048,
        num_prefix_tokens: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_prefix_tokens = num_prefix_tokens
        self.llm_hidden_size = llm_hidden_size
        
        # 可学习的查询向量
        self.queries = nn.Parameter(torch.randn(1, num_prefix_tokens, llm_hidden_size) * 0.02)
        
        # 将ASM embedding扩展到序列
        self.asm_expand = nn.Linear(clap_hidden_size, llm_hidden_size)
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=llm_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size * 4, llm_hidden_size)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(llm_hidden_size)
        self.norm2 = nn.LayerNorm(llm_hidden_size)
        
    def forward(self, asm_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            asm_embedding: [batch_size, clap_hidden_size]
            
        Returns:
            prefix_tokens: [batch_size, num_prefix_tokens, llm_hidden_size]
        """
        batch_size = asm_embedding.size(0)
        
        # 扩展查询
        queries = self.queries.expand(batch_size, -1, -1)
        
        # 扩展ASM embedding作为key/value
        asm_expanded = self.asm_expand(asm_embedding).unsqueeze(1)  # [batch, 1, llm_hidden]
        
        # 交叉注意力
        attended, _ = self.cross_attention(queries, asm_expanded, asm_expanded)
        attended = self.norm1(queries + attended)
        
        # FFN
        output = self.norm2(attended + self.ffn(attended))
        
        return output


class GatedProjection(nn.Module):
    """
    带门控机制的投影层
    
    使用门控机制控制信息流，可能有助于训练稳定性
    """
    
    def __init__(
        self,
        clap_hidden_size: int = 768,
        llm_hidden_size: int = 2048,
        num_prefix_tokens: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_prefix_tokens = num_prefix_tokens
        self.llm_hidden_size = llm_hidden_size
        
        # 基础投影
        self.base_projection = nn.Linear(clap_hidden_size, num_prefix_tokens * llm_hidden_size)
        
        # 门控投影
        self.gate_projection = nn.Linear(clap_hidden_size, num_prefix_tokens * llm_hidden_size)
        
        # 可学习的偏置（类似于prefix tuning的思路）
        self.prefix_bias = nn.Parameter(torch.zeros(1, num_prefix_tokens, llm_hidden_size))
        
        self.layer_norm = nn.LayerNorm(llm_hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.base_projection.weight, std=0.02)
        nn.init.zeros_(self.base_projection.bias)
        nn.init.normal_(self.gate_projection.weight, std=0.02)
        nn.init.zeros_(self.gate_projection.bias)
        
    def forward(self, asm_embedding: torch.Tensor) -> torch.Tensor:
        batch_size = asm_embedding.size(0)
        
        # 基础投影
        base = self.base_projection(asm_embedding)
        base = base.view(batch_size, self.num_prefix_tokens, self.llm_hidden_size)
        
        # 门控
        gate = torch.sigmoid(self.gate_projection(asm_embedding))
        gate = gate.view(batch_size, self.num_prefix_tokens, self.llm_hidden_size)
        
        # 门控融合
        output = gate * base + (1 - gate) * self.prefix_bias.expand(batch_size, -1, -1)
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


def create_projection(
    projection_type: str = "mlp",
    **kwargs
) -> nn.Module:
    """
    工厂函数：创建投影层
    
    Args:
        projection_type: "mlp", "cross_attention", "gated"
        **kwargs: 传递给具体投影层的参数
        
    Returns:
        投影层实例
    """
    if projection_type == "mlp":
        return AsmProjection(**kwargs)
    elif projection_type == "cross_attention":
        return CrossAttentionProjection(**kwargs)
    elif projection_type == "gated":
        return GatedProjection(**kwargs)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")
