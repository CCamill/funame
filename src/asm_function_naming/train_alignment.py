"""
阶段1：对比学习对齐训练

将CLAP-ASM的表示空间与Qwen的中间层表示空间对齐
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, "/home/lab314/cjw/funame")
from config import get_model_config, get_training_config
from models.asm_naming_model import AsmNamingModelForAlignment
from dataset import AsmAlignmentDataset, create_dataloaders
from utils import (
    set_seed, setup_logging, save_config,
    AverageMeter, EarlyStopping, count_parameters,
    print_gpu_memory, create_optimizer
)


def train_one_epoch(
    model: AsmNamingModelForAlignment,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
    logger
):
    """训练一个epoch"""
    model.train()
    # 注意：Qwen是冻结的，只有CLAP encoder和投影层在训练
    model.clap_encoder.train()
    model.clap_projection.train()
    model.qwen_projection.train()
    
    loss_meter = AverageMeter()
    pos_sim_meter = AverageMeter()
    neg_sim_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # 移动数据到设备
        asm_input_ids = batch["asm_input_ids"].to(device)
        asm_attention_mask = batch["asm_attention_mask"].to(device)
        src_input_ids = batch["src_input_ids"].to(device)
        src_attention_mask = batch["src_attention_mask"].to(device)
        
        # 前向传播
        outputs = model(
            asm_input_ids=asm_input_ids,
            asm_attention_mask=asm_attention_mask,
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask
        )
        
        loss = outputs["loss"]
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        
        optimizer.step()
        scheduler.step()
        
        # 更新统计
        batch_size = asm_input_ids.size(0)
        loss_meter.update(loss.item(), batch_size)
        pos_sim_meter.update(outputs["pos_similarity"].item(), batch_size)
        neg_sim_meter.update(outputs["neg_similarity"].item(), batch_size)
        
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "pos_sim": f"{pos_sim_meter.avg:.4f}",
            "neg_sim": f"{neg_sim_meter.avg:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
    logger.info(
        f"Epoch {epoch} - Loss: {loss_meter.avg:.4f}, "
        f"Pos Sim: {pos_sim_meter.avg:.4f}, Neg Sim: {neg_sim_meter.avg:.4f}"
    )
    
    return loss_meter.avg, pos_sim_meter.avg, neg_sim_meter.avg


@torch.no_grad()
def evaluate(
    model: AsmNamingModelForAlignment,
    val_loader: DataLoader,
    device: str,
    logger
):
    """验证"""
    model.eval()
    
    loss_meter = AverageMeter()
    pos_sim_meter = AverageMeter()
    neg_sim_meter = AverageMeter()
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        asm_input_ids = batch["asm_input_ids"].to(device)
        asm_attention_mask = batch["asm_attention_mask"].to(device)
        src_input_ids = batch["src_input_ids"].to(device)
        src_attention_mask = batch["src_attention_mask"].to(device)
        
        outputs = model(
            asm_input_ids=asm_input_ids,
            asm_attention_mask=asm_attention_mask,
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask
        )
        
        batch_size = asm_input_ids.size(0)
        loss_meter.update(outputs["loss"].item(), batch_size)
        pos_sim_meter.update(outputs["pos_similarity"].item(), batch_size)
        neg_sim_meter.update(outputs["neg_similarity"].item(), batch_size)
        
    logger.info(
        f"Validation - Loss: {loss_meter.avg:.4f}, "
        f"Pos Sim: {pos_sim_meter.avg:.4f}, Neg Sim: {neg_sim_meter.avg:.4f}"
    )
    
    return loss_meter.avg, pos_sim_meter.avg, neg_sim_meter.avg


def main(args):
    # 配置
    model_config = get_model_config()
    train_config = get_training_config()
    
    # 覆盖默认配置
    if args.data_path:
        train_config.data_path = args.data_path
    if args.output_dir:
        train_config.output_dir = args.output_dir
    if args.epochs:
        train_config.alignment_epochs = args.epochs
    if args.batch_size:
        train_config.alignment_batch_size = args.batch_size
    if args.lr:
        train_config.alignment_lr = args.lr
        
    # 设置
    set_seed(train_config.seed)
    device = train_config.device
    output_dir = os.path.join(train_config.output_dir, "alignment")
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir, "alignment_training")
    save_config(train_config, output_dir, "training_config.json")
    save_config(model_config, output_dir, "model_config.json")
    
    logger.info(f"Training config: {train_config}")
    logger.info(f"Model config: {model_config}")
    
    # 创建模型
    logger.info("Creating model...")
    model = AsmNamingModelForAlignment(
        clap_model_name=model_config.clap_asm_model_name,
        qwen_model_name=model_config.qwen_model_name,
        clap_hidden_size=model_config.clap_asm_hidden_size,
        qwen_hidden_size=model_config.qwen_hidden_size,
        projection_dim=512,
        use_4bit=model_config.use_4bit,
        device=device
    )
    
    logger.info(f"Trainable parameters: {count_parameters(model):,}")
    print_gpu_memory()
    
    # 创建数据加载器
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_path=train_config.data_path,
        clap_tokenizer=model.clap_encoder.tokenizer,
        qwen_tokenizer=model.qwen_tokenizer,
        batch_size=train_config.alignment_batch_size,
        train_split=train_config.train_split,
        dataset_type="alignment",
        max_asm_length=train_config.max_asm_length,
        max_src_length=train_config.max_src_length
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
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
    
    num_training_steps = len(train_loader) * train_config.alignment_epochs
    num_warmup_steps = int(num_training_steps * train_config.alignment_warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 早停
    early_stopping = EarlyStopping(patience=10, mode="max")  # 监控正样本相似度
    
    # 训练循环
    best_pos_sim = -1
    
    for epoch in range(1, train_config.alignment_epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{train_config.alignment_epochs}")
        logger.info(f"{'='*50}")
        
        # 训练
        train_loss, train_pos_sim, train_neg_sim = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, logger
        )
        
        # 验证
        val_loss, val_pos_sim, val_neg_sim = evaluate(
            model, val_loader, device, logger
        )
        
        # 保存最佳模型
        if val_pos_sim > best_pos_sim:
            best_pos_sim = val_pos_sim
            save_path = os.path.join(output_dir, "best_alignment_model")
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
                model.qwen_projection.state_dict(),
                os.path.join(save_path, "qwen_projection.pt")
            )
            
            logger.info(f"Saved best model with pos_sim: {best_pos_sim:.4f}")
            
        # 早停检查
        if early_stopping(val_pos_sim):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
            
        print_gpu_memory()
        
    logger.info(f"\nTraining completed! Best pos_sim: {best_pos_sim:.4f}")
    logger.info(f"Model saved to {output_dir}")
    
    # 返回对齐后的模型路径
    return os.path.join(output_dir, "best_alignment_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Contrastive Learning Alignment")
    parser.add_argument("--data_path", type=str, help="Path to CSV data file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    
    args = parser.parse_args()
    main(args)
