import os 
import sys 

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup
)
from tqdm import tqdm
from datetime import datetime

from utils import setup_logger
from dataplay import FuncAbstractDataset
from loss_function import InfoNCELoss


logger = setup_logger('train_contrastive')

@dataclass
class TrainConfig:
    dataset_path: str = r"resources/dataset/func_pairs_with_abstract.csv"
    epochs: int = 150
    lr: float = 1e-5
    batch_size: int = 32
    weight_decay: float = 0.01
    warmup_steps: int = 500
    projection_dim: int = 768  # 投影层输出维度
    model_save_dir: str = f'resources/finetuning-{datetime.date}/'
    

class ProjectionHead(nn.Module):
    """投影头，用于将不同维度的embedding映射到相同空间"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 4 * output_dim),
            nn.ReLU(),
            nn.Linear(4 * output_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class AsmAbstractContrastive(nn.Module):
    def __init__(self, asm_model, abstract_model, config, train_dataloader, test_dataloader, eval_dataloader):
        super().__init__()
        self.config = config
        
        self.asm_model = asm_model
        self.abstract_model = abstract_model
        
        # 获取两个模型的输出维度
        self.asm_dim = asm_model.config.hidden_size
        self.abstract_dim = abstract_model.config.hidden_size
        
        # 创建投影头（如果维度不同）
        if self.asm_dim != self.abstract_dim or config.projection_dim:
            proj_dim = config.projection_dim if config.projection_dim else max(self.asm_dim, self.abstract_dim)
            self.asm_projection = ProjectionHead(self.asm_dim, proj_dim)
            self.abstract_projection = ProjectionHead(self.abstract_dim, proj_dim)
        else:
            self.asm_projection = None
            self.abstract_projection = None
        
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        
        self.loss_fn = InfoNCELoss()
        self.optimizer, self.scheduler = self._create_optimizer_and_scheduler()
        
        # 移动到GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        logger.info(f"使用设备: {self.device}")
    
    def _create_optimizer_and_scheduler(self):
        """创建优化器和学习率调度器"""
        # 分组参数：分别为两个编码器和投影头设置不同学习率
        param_groups = []
        
        # ASM模型参数（较小学习率）
        asm_params = [p for p in self.asm_model.parameters() if p.requires_grad]
        if asm_params:
            param_groups.append({
                "params": asm_params,
                "lr": self.config.lr,
                "name": "asm_encoder"
            })
        
        # Abstract模型参数（较小学习率）
        abstract_params = [p for p in self.abstract_model.parameters() if p.requires_grad]
        if abstract_params:
            param_groups.append({
                "params": abstract_params,
                "lr": self.config.lr,
                "name": "abstract_encoder"
            })
        
        # 投影头参数（较大学习率）
        if self.asm_projection is not None:
            projection_params = list(self.asm_projection.parameters()) + \
                              list(self.abstract_projection.parameters())
            param_groups.append({
                "params": projection_params,
                "lr": self.config.lr * 10,  # 投影头使用10倍学习率
                "name": "projections"
            })
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )
        
        # 创建学习率调度器
        total_steps = len(self.train_dataloader) * self.config.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def get_embeddings(self, asm_inputs, abstract_inputs):
        """获取两个模型的embedding"""
        # ASM模型前向传播
        asm_embeddings = self.asm_model(**asm_inputs)  # [batch, 768],  clap-asm 默认使用mean pooling 
        
        # Abstract模型前向传播
        abstract_outputs = self.abstract_model(**abstract_inputs)   # sentence-bert一般使用mean pooling的方式
        if hasattr(abstract_outputs, 'pooler_output') and abstract_outputs.pooler_output is not None:
            abstract_embeddings = abstract_outputs.pooler_output
        else:
            abstract_embeddings = abstract_outputs.last_hidden_state[:, 0, :]
        
        # 应用投影头
        if self.asm_projection is not None:
            asm_embeddings = self.asm_projection(asm_embeddings)
            abstract_embeddings = self.abstract_projection(abstract_embeddings)
        
        # L2归一化
        asm_embeddings = nn.functional.normalize(asm_embeddings, p=2, dim=1)
        abstract_embeddings = nn.functional.normalize(abstract_embeddings, p=2, dim=1)
        
        return asm_embeddings, abstract_embeddings
    
    def train_model(self):
        """训练模型"""
        logger.info("开始训练...")
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # 训练阶段
            self.train()
            train_loss = 0.0
            train_metrics = {}
            train_bar = tqdm(enumerate(self.train_dataloader), 
                                total=len(self.train_dataloader),
                                desc=f"Epoch {epoch+1}/{self.config.epochs}",
                                ncols=100)
            for batch_idx, batch in train_bar:
                # 将数据移到设备上
                asm_inputs = {k: v.to(self.device) for k, v in batch['asm_inputs'].items()}
                abstract_inputs = {k: v.to(self.device) for k, v in batch['abstract_inputs'].items()}
                
                # 前向传播
                asm_embeddings, abstract_embeddings = self.get_embeddings(asm_inputs, abstract_inputs)
                
                # 计算损失
                loss, metrics = self.loss_fn(asm_embeddings, abstract_embeddings)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                train_loss += loss.item()
                for k, v in metrics.items():
                    train_metrics[k] = train_metrics.get(k, 0) + v
                
                # 日志记录
                if batch_idx % 10 == 0:
                    train_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                    })
            
            # 计算平均训练损失
            avg_train_loss = train_loss / len(self.train_dataloader)
            avg_train_metrics = {k: v / len(self.train_dataloader) for k, v in train_metrics.items()}
            
            logger.info(f"Epoch {epoch+1} - 训练完成")
            logger.info(f"平均训练损失: {avg_train_loss:.4f}")
            logger.info(f"训练指标: {avg_train_metrics}")
            
            # 评估阶段
            eval_loss, eval_metrics = self.evaluate()
            logger.info(f"验证损失: {eval_loss:.4f}")
            logger.info(f"验证指标: {eval_metrics}\n")
            
            # 保存最佳模型
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_model(f'{self.config.model_save_dir}/best_model_epoch_{epoch+1}.pt')
                logger.info(f"保存最佳模型 (验证损失: {eval_loss:.4f})\n")
    
    def evaluate(self):
        """评估模型"""
        self.eval()
        total_loss = 0.0
        total_metrics = {}
        
        with torch.no_grad():
            eval_bar = tqdm(self.eval_dataloader, 
                            total=len(self.eval_dataloader),
                            desc="Evaluating",
                            ncols=100)
            for batch in eval_bar:
                asm_inputs = {k: v.to(self.device) for k, v in batch['asm_inputs'].items()}
                abstract_inputs = {k: v.to(self.device) for k, v in batch['abstract_inputs'].items()}
                
                asm_embeddings, abstract_embeddings = self.get_embeddings(asm_inputs, abstract_inputs)
                loss, metrics = self.loss_fn(asm_embeddings, abstract_embeddings)
                
                total_loss += loss.item()
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
        
        avg_loss = total_loss / len(self.eval_dataloader)
        avg_metrics = {k: v / len(self.eval_dataloader) for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'asm_model_state_dict': self.asm_model.state_dict(),
            'abstract_model_state_dict': self.abstract_model.state_dict(),
            'asm_projection_state_dict': self.asm_projection.state_dict() if self.asm_projection else None,
            'abstract_projection_state_dict': self.abstract_projection.state_dict() if self.abstract_projection else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"模型已保存到 {path}")


def split_dataset_sklearn_sequential(dataset_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """使用sklearn划分数据集，保持原始顺序"""
    df = pd.read_csv(dataset_path)
    total_samples = len(df)
    
    logger.info(f"数据集总样本数: {total_samples}")
    
    indices = np.arange(total_samples)
    
    # 第一次划分：训练集 vs (验证集+测试集)
    train_indices, temp_indices = train_test_split(
        indices, 
        train_size=train_ratio,
        shuffle=False,
        random_state=None
    )
    
    # 第二次划分：验证集 vs 测试集
    val_ratio_relative = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio_relative,
        shuffle=False,
        random_state=None
    )
    
    train_set = df.iloc[train_indices]
    val_set = df.iloc[val_indices]
    test_set = df.iloc[test_indices]
    
    logger.info(f"训练集: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")
    
    return train_set, val_set, test_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Contrastive Learning for Code-Abstract Alignment")
    parser.add_argument("--asm_model_name", type=str, default='hustcw/clap-asm', 
                       help='ASM/Code embedding model')
    parser.add_argument("--text_model_name", type=str, 
                       default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                       help='Text embedding model')
    parser.add_argument("--model_save_dir", type=str, default=f'resources/finetuning-{datetime.now}/', help='模型保存目录')
    parser.add_argument("--epoch", type=int, default=100, help='训练轮数')
    parser.add_argument("--lr", type=float, default=1e-5, help='学习率')
    parser.add_argument("--batch_size", type=int, default=32, help='批次大小')
    parser.add_argument("--data_path", type=str, 
                       default='resources/dataset/func_pairs_with_abstract.csv',
                       help='数据集路径')
    parser.add_argument("--weight_decay", type=float, default=0.01, help='权重衰减')
    parser.add_argument("--projection_dim", type=int, default=256, help='投影维度')
    parser.add_argument("--warmup_steps", type=int, default=500, help='预热步数')
    
    args = parser.parse_args()
    
    # 配置
    config = TrainConfig(
        dataset_path=args.data_path,
        model_save_dir = args.model_save_dir,
        epochs=args.epoch,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        projection_dim=args.projection_dim,
        warmup_steps=args.warmup_steps
    )
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    
    logger.info("加载模型...")
    # 加载tokenizers和models
    asm_tokenizer = AutoTokenizer.from_pretrained(args.asm_model_name, trust_remote_code=True)
    asm_model = AutoModel.from_pretrained(args.asm_model_name, trust_remote_code=True)
    
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    abstract_model = AutoModel.from_pretrained(args.text_model_name)
    
    logger.info("加载数据集...")
    # 划分数据集
    train_set, val_set, test_set = split_dataset_sklearn_sequential(args.data_path)
    
    # 创建数据加载器
    train_dataset = FuncAbstractDataset(train_set, asm_tokenizer, text_tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    eval_dataset = FuncAbstractDataset(val_set, asm_tokenizer, text_tokenizer)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    test_dataset = FuncAbstractDataset(test_set, asm_tokenizer, text_tokenizer)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    # 创建模型并训练
    model = AsmAbstractContrastive(
        asm_model, 
        abstract_model, 
        config, 
        train_dataloader, 
        eval_dataloader, 
        test_dataloader
    )
    
    model.train_model()