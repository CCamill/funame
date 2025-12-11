"""
工具函数
"""
import torch
import random
import numpy as np
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def setup_logging(output_dir: str, log_name: str = "training"):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_config(config: Any, output_dir: str, filename: str = "config.json"):
    """保存配置到JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    config_dict = {}
    if hasattr(config, "__dict__"):
        config_dict = config.__dict__
    elif hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict
        config_dict = asdict(config)
    else:
        config_dict = dict(config)
        
    with open(os.path.join(output_dir, filename), "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
        

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置"""
    with open(config_path, "r") as f:
        return json.load(f)
    

class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停机制"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == "min":
            is_improvement = score < self.best_score - self.min_delta
        else:
            is_improvement = score > self.best_score + self.min_delta
            
        if is_improvement:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """计算模型参数数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_info() -> Dict[str, float]:
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated
        }
    return {}


def print_gpu_memory():
    """打印GPU内存使用情况"""
    info = get_gpu_memory_info()
    if info:
        print(f"GPU Memory - Allocated: {info['allocated_gb']:.2f}GB, "
              f"Reserved: {info['reserved_gb']:.2f}GB, "
              f"Max: {info['max_allocated_gb']:.2f}GB")


def compute_metrics(
    predictions: list,
    references: list
) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        predictions: 预测的函数名列表
        references: 真实的函数名列表
        
    Returns:
        包含各种指标的字典
    """
    assert len(predictions) == len(references)
    
    # 精确匹配
    exact_match = sum(p.lower() == r.lower() for p, r in zip(predictions, references)) / len(predictions)
    
    # 前缀匹配
    prefix_match = sum(
        r.lower().startswith(p.lower()) or p.lower().startswith(r.lower())
        for p, r in zip(predictions, references)
    ) / len(predictions)
    
    # 编辑距离（归一化）
    def normalized_edit_distance(s1, s2):
        s1, s2 = s1.lower(), s2.lower()
        if len(s1) == 0 and len(s2) == 0:
            return 0.0
        
        # 动态规划计算编辑距离
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    
        return dp[m][n] / max(m, n)
    
    avg_edit_distance = sum(
        normalized_edit_distance(p, r)
        for p, r in zip(predictions, references)
    ) / len(predictions)
    
    # 字符级F1
    def char_f1(pred, ref):
        pred_chars = set(pred.lower())
        ref_chars = set(ref.lower())
        
        if len(pred_chars) == 0 or len(ref_chars) == 0:
            return 0.0
            
        common = pred_chars & ref_chars
        precision = len(common) / len(pred_chars) if pred_chars else 0
        recall = len(common) / len(ref_chars) if ref_chars else 0
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    avg_char_f1 = sum(
        char_f1(p, r)
        for p, r in zip(predictions, references)
    ) / len(predictions)
    
    return {
        "exact_match": exact_match,
        "prefix_match": prefix_match,
        "avg_edit_distance": avg_edit_distance,
        "avg_char_f1": avg_char_f1
    }


def create_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    no_decay_keywords: list = None
) -> torch.optim.Optimizer:
    """
    创建优化器，对特定参数不使用weight decay
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay_keywords)
            ],
            "weight_decay": weight_decay
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay_keywords)
            ],
            "weight_decay": 0.0
        }
    ]
    
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """
    创建带warmup的线性学习率调度器
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
