"""
阶段1：对比学习对齐训练

将CLAP-ASM的表示空间与Qwen的中间层表示空间对齐
"""
import os
import sys
import argparse
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from datetime import datetime
import numpy as np
import torch.nn.functional as F
import json

sys.path.insert(0, "/home/lab314/cjw/funame")
from config import get_model_config, get_training_config, ModelConfig, TrainingConfig
from models.asm_naming_model import AsmNamingModelForAlignment
from dataset import AsmAlignmentDataset, create_dataloaders
from asm_function_naming_utils import (
    set_seed, setup_logger, save_config,
    AverageMeter, EarlyStopping, count_parameters,
    print_gpu_memory
)

logger = setup_logger(f"alignment_training")

def train_one_epoch(
    model: AsmNamingModelForAlignment,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
    logger,
    gradient_accumulation_steps: int = 1,
    logging_steps: int = 10
):
    """训练一个epoch"""
    model.train()
    # 注意：Source model是冻结的，只有CLAP encoder和投影层在训练
    model.clap_encoder.train()
    model.clap_projection.train()
    model.source_projection.train()

    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    total_loss = 0.0
    for step, batch in enumerate(pbar):
        # 移动数据到设备
        asm_input_ids = batch["asm_input_ids"].to(device)
        asm_attention_mask = batch["asm_attention_mask"].to(device)
        src_input_ids = batch["src_input_ids"].to(device)
        src_attention_mask = batch["src_attention_mask"].to(device)
        
        # 前向传播
        temperature = max(0.05 - 0.0005*epoch, 0.008)
        loss, metrics = model(
            asm_input_ids=asm_input_ids,
            asm_attention_mask=asm_attention_mask,
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            temperature=temperature
        )
        
        loss = loss / gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        total_loss += loss.item() * gradient_accumulation_steps
        
        # 梯度累积
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            # 日志记录
            if step % logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
    
    # 处理最后一个不完整的梯度累积
    if len(train_loader) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
    
    return total_loss / (step + 1), metrics
    


def compute_retrieval_metrics(
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """计算检索指标"""
        # L2归一化
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        key_embeddings = F.normalize(key_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(query_embeddings, key_embeddings.T)
        
        # 获取排名
        # 对每个query，按相似度降序排列，找到正确匹配（对角线）的位置
        num_samples = query_embeddings.shape[0]
        ranks = []
        
        for i in range(num_samples):
            sims = similarity_matrix[i]
            # 按相似度降序排列
            sorted_indices = torch.argsort(sims, descending=True)
            # 找到正确答案（索引i）的位置
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        # 计算指标
        mrr = np.mean(1.0 / ranks)
        recall_at_1 = np.mean(ranks <= 1)
        recall_at_5 = np.mean(ranks <= 5)
        recall_at_10 = np.mean(ranks <= 10)
        recall_at_100 = np.mean(ranks <= 100)
        
        return {
            'mrr': mrr,
            'recall@1': recall_at_1,
            'recall@5': recall_at_5,
            'recall@10': recall_at_10,
            'recall@100': recall_at_100,
            'mean_rank': np.mean(ranks)
        }


@torch.no_grad()
def evaluate(
    model: AsmNamingModelForAlignment,
    val_loader: DataLoader,
    device: str
):
    """验证"""
    model.eval()
    all_asm_embeds = []
    all_src_embeds = []
    for batch in tqdm(val_loader, desc="Evaluating"):
        asm_input_ids = batch["asm_input_ids"].to(device)
        asm_attention_mask = batch["asm_attention_mask"].to(device)
        src_input_ids = batch["src_input_ids"].to(device)
        src_attention_mask = batch["src_attention_mask"].to(device)
        
        asm_embeds = model.encode_asm(asm_input_ids, asm_attention_mask)
        src_embeds = model.encode_src(src_input_ids, src_attention_mask)
        
        all_asm_embeds.append(asm_embeds.cpu())
        all_src_embeds.append(src_embeds.cpu())
        
    all_asm_embeds = torch.cat(all_asm_embeds, dim=0)
    all_src_embeds = torch.cat(all_src_embeds, dim=0)
    
    metrics = compute_retrieval_metrics(all_asm_embeds, all_src_embeds)
    
    return metrics

def save_checkpoint(model: AsmNamingModelForAlignment, name:str, output_dir: str, epoch: int, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, best_eval_mrr: float):
    """保存模型"""
    save_path = os.path.join(output_dir, name)
    os.makedirs(save_path, exist_ok=True)
    
    # 保存CLAP encoder和投影层
    torch.save(
        model.clap_encoder.state_dict(),
        os.path.join(save_path, "clap_encoder.pt")
    )
    torch.save(
        model.clap_projection.state_dict(),
        os.path.join(save_path, "clap_projection.pt")
    )
    torch.save(
        model.source_projection.state_dict(),
        os.path.join(save_path, "source_projection.pt")
    )

    # 保存训练状态和优化器和调度器 用于恢复训练
    state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_eval_mrr': best_eval_mrr
    }
    torch.save(state, os.path.join(save_path, "training_state.pt"))


def main(model_config, train_config):
    # 设置
    set_seed(train_config.seed)
    device = train_config.device
    output_dir = train_config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 创建模型
    model = AsmNamingModelForAlignment(
        clap_model_name=model_config.clap_asm_model_name,
        src_model_name=model_config.src_model_name,
        clap_hidden_size=model_config.clap_asm_hidden_size,
        src_hidden_size=model_config.src_hidden_size,
        projection_dim=model_config.projection_dim,
        use_4bit=model_config.use_4bit,
        device=device,
        load_checkpoint=args.load_checkpoint
    )
    
    logger.info(f"Trainable parameters: {count_parameters(model) / 1000000:.2f}M")
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        train_data_path=train_config.train_data_path,
        eval_data_path=train_config.eval_data_path,
        clap_tokenizer=model.clap_encoder.tokenizer,
        src_tokenizer=model.source_tokenizer,
        train_batch_size=train_config.alignment_train_batch_size,
        eval_batch_size=train_config.alignment_eval_batch_size,
        dataset_type="alignment",
        max_asm_length=train_config.max_asm_length,
        max_src_length=train_config.max_src_length,
        num_workers=train_config.num_workers,
        logger=logger
    )

    
    # 优化器和调度器
    # 只优化可训练参数
    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config.alignment_lr,
        weight_decay=0.01
    )
    
    # 考虑梯度累积，实际训练步数会减少
    gradient_accumulation_steps = train_config.alignment_gradient_accumulation_steps
    num_training_steps = (len(train_loader) // gradient_accumulation_steps) * train_config.alignment_epochs
    num_warmup_steps = int(num_training_steps * train_config.alignment_warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    if args.load_checkpoint:
        training_state = torch.load(os.path.join(args.load_checkpoint, "training_state.pt"), weights_only=False)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        scheduler.load_state_dict(training_state["scheduler_state_dict"])
        best_eval_mrr = training_state["best_eval_mrr"]
        start_epoch = training_state["epoch"] + 1
        logger.info(f"Loaded checkpoint from {args.load_checkpoint}")
        logger.info(f"Best MRR: {best_eval_mrr:.4f}, starting from epoch {start_epoch}")
    else:
        best_eval_mrr = -1.0
        start_epoch = 1
    for epoch in range(start_epoch, train_config.alignment_epochs + 1):
        logger.info(f"{'-'*30}")
        logger.info(f"Epoch {epoch}/{train_config.alignment_epochs}")
        
        # 训练
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, logger,
            gradient_accumulation_steps=train_config.alignment_gradient_accumulation_steps,
        )
        logger.info(
            f"Epoch {epoch} completed. loss: {train_loss:.4f}, mean_pos_sim:{train_metrics['mean_pos_sim']:.4f}, mean_neg_sim:{train_metrics['mean_neg_sim']:.4f}"
        )
        # 验证
        eval_metrics = evaluate(
            model, val_loader, device
        )

        logger.info(
                f"Eval MRR: {eval_metrics['mrr']:.4f}, "
                f"Recall@1: {eval_metrics['recall@1']:.4f}, "
                f"Recall@5: {eval_metrics['recall@5']:.4f}, "
                f"Recall@10: {eval_metrics['recall@10']:.4f}, "
                f"Recall@100: {eval_metrics['recall@100']:.4f}, "
                f"Mean Rank: {eval_metrics['mean_rank']:.2f}"
                )
        
        save_checkpoint(model, f"Epoch_{epoch}", output_dir, epoch, optimizer, scheduler, best_eval_mrr)
        
        # 保存最佳模型
        if eval_metrics['mrr'] > best_eval_mrr:
            best_eval_mrr = eval_metrics['mrr']
            save_checkpoint(model, "best", output_dir, epoch, optimizer, scheduler, best_eval_mrr)
            logger.info(f"New best MRR: {best_eval_mrr:.4f}, checkpoint saved")
        
    logger.info(f"\nTraining completed!")
    logger.info(f"Best MRR: {best_eval_mrr:.4f}, checkpoint saved to {output_dir}/best")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Contrastive Learning Alignment")
    parser.add_argument("--clap_model_name", type=str, default="hustcw/clap-asm", help="CLAP model name")
    parser.add_argument("--src_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Source model name", choices=["Qwen/Qwen2.5-Coder-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"])
    parser.add_argument("--clap_hidden_size", type=int, default=768, help="CLAP hidden size")
    parser.add_argument("--src_hidden_size", type=int, default=2048, help="Source hidden size", choices=[2048, 4096])
    parser.add_argument("--projection_dim", type=int, default=768, help="Projection dimension")
    parser.add_argument("--use_4bit", type=bool, default=True, help="Use 4bit quantization")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--max_asm_length", type=int, default=1024, help="Max assembly code length")
    parser.add_argument("--max_src_length", type=int, default=1024, help="Max source code length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_data_path", type=str, default="resources/sym-dataset/func_pairs_with_strings_train.csv", help="Path to CSV train data file")
    parser.add_argument("--eval_data_path", type=str, default="resources/sym-dataset/func_pairs_with_strings_eval.csv", help="Path to CSV eval data file")
    parser.add_argument("--output_dir", type=str, default=f"resources/asm-function-naming/alignment-{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load checkpoint example: resources/asm-function-naming/alignment-20251216_120407/Epoch_10")
    
    args = parser.parse_args()
    if args.src_model_name == "meta-llama/Llama-3.2-3B-Instruct":
        args.src_hidden_size = 3072
    else:
        args.src_hidden_size = 2048
    if args.load_checkpoint:
        args.output_dir = os.path.dirname(args.load_checkpoint)

    # 配置
    if args.load_checkpoint:
        model_config = ModelConfig.from_json(os.path.join(os.path.dirname(args.load_checkpoint), "model_config.json"))
        train_config = TrainingConfig.from_json(os.path.join(os.path.dirname(args.load_checkpoint), "training_config.json"))
        logger.info(f"Training continued from checkpoint {args.load_checkpoint}")
    else:
        model_config = ModelConfig(
            clap_asm_model_name=args.clap_model_name,
            src_model_name=args.src_model_name,
            clap_asm_hidden_size=args.clap_hidden_size,
            src_hidden_size=args.src_hidden_size,
            use_4bit=args.use_4bit,
            projection_dim=args.projection_dim
        )
        train_config = TrainingConfig(
            train_data_path=args.train_data_path,
            eval_data_path=args.eval_data_path,
            max_asm_length=args.max_asm_length,
            max_src_length=args.max_src_length,
            alignment_epochs=args.epochs,
            alignment_train_batch_size=args.train_batch_size,
            alignment_eval_batch_size=args.eval_batch_size,
            alignment_lr=args.lr,
            alignment_gradient_accumulation_steps=args.gradient_accumulation_steps,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
            num_workers=args.num_workers
        )
        logger.info(f"Training started from scratch")
    logger.info(f"asm_model_name: {model_config.clap_asm_model_name}")
    logger.info(f"src_model_name: {model_config.src_model_name}")
    logger.info(f"clap_hidden_size: {model_config.clap_asm_hidden_size}")
    logger.info(f"src_hidden_size: {model_config.src_hidden_size}")
    logger.info(f"projection_dim: {model_config.projection_dim}")
    logger.info(f"use_4bit: {model_config.use_4bit}")
    logger.info(f"device: {train_config.device}")
    logger.info(f"num_workers: {train_config.num_workers}")
    logger.info(f"train_data_path: {train_config.train_data_path}")
    logger.info(f"eval_data_path: {train_config.eval_data_path}")
    logger.info(f"output_dir: {train_config.output_dir}")
    logger.info(f"epochs: {train_config.alignment_epochs}")
    logger.info(f"train_batch_size: {train_config.alignment_train_batch_size}")
    logger.info(f"eval_batch_size: {train_config.alignment_eval_batch_size}")
    logger.info(f"lr: {train_config.alignment_lr}")

    if not args.load_checkpoint:
        save_config(train_config, train_config.output_dir, "training_config.json")
        save_config(model_config, train_config.output_dir, "model_config.json")

    main(model_config, train_config)
