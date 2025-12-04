# Load model directly
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup
)

from dataplay import FuncAbstractDataset
from loss_function import InfoNCELoss


@dataclass
class TrainConfig:
    dataset_path:str = r"resources/dataset/func_pairs_with_abstract.csv"
    epochs: int = 150
    lr: float = 1e-5
    batch_size: int = 32
    weight_decay: float =0.01


class AsmAbstractContrastive():
    def __init__(self, asm_model, abstract_model, config, train_dataloader, test_dataloader, eval_dataloader):
        self.config = config
        
        self.asm_model = asm_model
        self.abstract_model = abstract_model
        
        self.loss_fn = InfoNCELoss()
        self.optimizer = self._create_optimizer()
            
    def _create_optimizer(self):
        base_params = []
        projection_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lm_head' in name:
                    projection_params.append(param)
                else:
                    base_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {"params": base_params, "lr": self.config.lr},
            {"params": projection_params, 'lr': self.config.lr * 10}    # 投影头使用更大学习率
        ], weight_decay=self.config.weight_decay)
        
        return optimizer
    def train(self):
        for epoch in self.config.epochs:
            for batch_idx, batch in enumerate(self.train_dataloader):
                asm_inputs = batch['asm_inputs']
                abstract_inputs = batch['abstract_inputs']
                asm_embeddings = self.asm_model(asm_inputs)
                abstract_embeddings = self.abstract_model(abstract_inputs)
                loss, metrics = self.loss_fn(asm_embeddings, abstract_embeddings)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}, Metrics: {metrics}")
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.eval_dataloader):
                    asm_inputs = batch['asm_inputs']
                    abstract_inputs = batch['abstract_inputs']
                    asm_embeddings = self.asm_model(**asm_inputs)
                    abstract_embeddings = self.abstract_model(**abstract_inputs)
                    loss, metrics = self.loss_fn(asm_embeddings, abstract_embeddings)
                    print(f"Epoch {epoch}, Batch {batch_idx}, Eval Loss: {loss.item()}, Eval Metrics: {metrics}")


def split_dataset_sklearn_sequential(self, dataset_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    使用sklearn划分数据集，保持原始顺序
    """
    # 读取数据
    df = pd.read_csv(dataset_path)
    total_samples = len(df)
    
    print(f"数据集总样本数: {total_samples}")
    
    # 创建索引数组以保持顺序信息
    indices = np.arange(total_samples)
    
    # 第一次划分：训练集 vs (验证集+测试集)
    train_indices, temp_indices = train_test_split(
        indices, 
        train_size=train_ratio,
        shuffle=False,  # 关键：关闭随机打乱
        random_state=None  # 不需要随机种子
    )
    
    # 第二次划分：验证集 vs 测试集
    val_ratio_relative = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio_relative,
        shuffle=False,  # 保持顺序
        random_state=None
    )
    
    # 根据索引获取数据
    train_set = df.iloc[train_indices]
    val_set = df.iloc[val_indices]
    test_set = df.iloc[test_indices]
    
    return train_set, val_set, test_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="contrastive")
    parser.add_argument("--asm_model_name", type=str, default='hustcw/clap-asm',  help='the path of asm embedding model')
    parser.add_argument("--asm_tokenizer", type=str, default='hustcw/clap-asm', help='the path of asm tokenizer')
    parser.add_argument("--test_model_name", type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  help='the path of text embedding model')
    parser.add_argument("--test_tokenizer", type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', help='the path of text tokenizer')
    parser.add_argument("--epoch", type=int, default=10, help='number of training epochs')
    parser.add_argument("--lr", type=float, default=1e-5, help='learning rate')
    parser.add_argument("--step_size", type=int, default=40000, help='scheduler step size')
    parser.add_argument("--batch_size", type=int, default =32, help='training batch size')
    parser.add_argument("--data_path", type=str, default='resources/dataset/func_pairs_with_abstract.csv', help='the path of training data')
    parser.add_argument("--weight_decay", type=int, default=0.01, help='the path of training data')
    args = parser.parse_args()
    
    config = TrainConfig(
        dataset_path=args.data_path,
        epochs=args.epoch,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay
    )
    asm_tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True)
    asm_model = AutoModel.from_pretrained("hustcw/clap-asm", trust_remote_code=True)

    text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    abstract_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    df = pd.read_csv(args.data_path)
    train_set, val_set, test_set = split_dataset_sklearn_sequential(args.data_path)
    train_dataset = FuncAbstractDataset(train_set, asm_tokenizer, text_tokenizer)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    eval_dataset = FuncAbstractDataset(val_set, asm_tokenizer, text_tokenizer)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    test_dataset = FuncAbstractDataset(test_set, asm_tokenizer, text_tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = AsmAbstractContrastive(asm_model, abstract_model, config, train_dataloader, eval_dataloader, test_dataloader)

    model.train()


        
        
