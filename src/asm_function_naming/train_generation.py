"""
阶段2：生成训练

训练完整的ASM函数命名模型
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_model_config, get_training_config
from models import AsmNamingModel
from dataset import AsmFunctionDataset, create_dataloaders
from utils import ( 
    set_seed, setup_logging, save_config,
    AverageMeter, EarlyStopping, count_parameters,
    print_gpu_memory, compute_metrics
)


def train_one_epoch(
    model: AsmNamingModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
    gradient_accumulation_steps: int,
    logger
):
    """训练一个epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        # 移动数据到设备
        asm_input_ids = batch["asm_input_ids"].to(device)
        asm_attention_mask = batch["asm_attention_mask"].to(device)
        prompt_input_ids = batch["prompt_input_ids"].to(device)
        prompt_attention_mask = batch["prompt_attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 前向传播
        outputs = model(
            asm_input_ids=asm_input_ids,
            asm_attention_mask=asm_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            labels=labels
        )
        
        loss = outputs["loss"]
        
        # 梯度累积
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        # 更新统计
        batch_size = asm_input_ids.size(0)
        loss_meter.update(loss.item() * gradient_accumulation_steps, batch_size)
        
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
    logger.info(f"Epoch {epoch} - Loss: {loss_meter.avg:.4f}")
    
    return loss_meter.avg


@torch.no_grad()
def evaluate(
    model: AsmNamingModel,
    val_loader: DataLoader,
    device: str,
    logger,
    num_samples: int = 50  # 用于生成评估的样本数
):
    """验证"""
    model.eval()
    
    loss_meter = AverageMeter()
    predictions = []
    references = []
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        asm_input_ids = batch["asm_input_ids"].to(device)
        asm_attention_mask = batch["asm_attention_mask"].to(device)
        prompt_input_ids = batch["prompt_input_ids"].to(device)
        prompt_attention_mask = batch["prompt_attention_mask"].to(device)
        labels = batch["labels"].to(device)
        func_names = batch["func_name"]
        
        # 计算loss
        outputs = model(
            asm_input_ids=asm_input_ids,
            asm_attention_mask=asm_attention_mask,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            labels=labels
        )
        
        batch_size = asm_input_ids.size(0)
        loss_meter.update(outputs["loss"].item(), batch_size)
        
        # 生成预测（只对部分样本做生成评估，节省时间）
        if len(predictions) < num_samples:
            for i in range(min(batch_size, num_samples - len(predictions))):
                # 解码ASM代码
                asm_code = model.clap_encoder.tokenizer.decode(
                    asm_input_ids[i],
                    skip_special_tokens=True
                )
                
                # 生成函数名
                pred_name = model.generate(
                    asm_code=asm_code,
                    max_new_tokens=32,
                    do_sample=False
                )
                
                predictions.append(pred_name)
                references.append(func_names[i])
                
    # 计算指标
    metrics = compute_metrics(predictions, references)
    
    logger.info(f"Validation - Loss: {loss_meter.avg:.4f}")
    logger.info(f"Validation - Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"Validation - Prefix Match: {metrics['prefix_match']:.4f}")
    logger.info(f"Validation - Char F1: {metrics['avg_char_f1']:.4f}")
    
    # 打印一些示例
    logger.info("\nSample predictions:")
    for i in range(min(5, len(predictions))):
        logger.info(f"  Pred: {predictions[i]}, Ref: {references[i]}")
        
    return loss_meter.avg, metrics


def main(args):
    # 配置
    model_config = get_model_config()
    train_config = get_training_config()
    
    # 覆盖默认配置
    if args.train_data_path:
        train_config.train_data_path = args.train_data_path
    if args.eval_data_path:
        train_config.eval_data_path = args.eval_data_path
    if args.output_dir:
        train_config.output_dir = args.output_dir
    if args.epochs:
        train_config.generation_epochs = args.epochs
    if args.batch_size:
        train_config.generation_train_batch_size = args.train_batch_size
    if args.eval_batch_size:
        train_config.generation_eval_batch_size = args.eval_batch_size
    if args.lr:
        train_config.generation_lr = args.lr
    if args.alignment_checkpoint:
        # 加载对齐阶段的权重
        pass  # 在模型创建后处理
        
    # 设置
    set_seed(train_config.seed)
    device = train_config.device
    output_dir = os.path.join(train_config.output_dir, "generation")
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir, "generation_training")
    save_config(train_config, output_dir, "training_config.json")
    save_config(model_config, output_dir, "model_config.json")
    
    logger.info(f"Training config: {train_config}")
    logger.info(f"Model config: {model_config}")
    
    # 创建模型
    logger.info("Creating model...")
    model = AsmNamingModel(
        clap_model_name=model_config.clap_asm_model_name,
        qwen_model_name=model_config.src_model_name,
        clap_hidden_size=model_config.clap_asm_hidden_size,
        qwen_hidden_size=model_config.src_hidden_size,
        num_prefix_tokens=model_config.num_prefix_tokens,
        projection_type="mlp",
        use_4bit=model_config.use_4bit,
        use_lora=True,
        lora_r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        lora_target_modules=model_config.lora_target_modules,
        freeze_clap=False,  # 可以继续微调CLAP
        device=device
    )
    
    # 加载对齐阶段的权重（如果有）
    if args.alignment_checkpoint and os.path.exists(args.alignment_checkpoint):
        logger.info(f"Loading alignment checkpoint from {args.alignment_checkpoint}")
        
        clap_state = torch.load(
            os.path.join(args.alignment_checkpoint, "clap_encoder.pt"),
            map_location=device
        )
        model.clap_encoder.load_state_dict(clap_state, strict=False)
        logger.info("Loaded CLAP encoder weights from alignment stage")
        
    logger.info(f"Trainable parameters: {count_parameters(model):,}")
    print_gpu_memory(logger)
    
    # 创建数据加载器
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_data_path=train_config.train_data_path,
        eval_data_path=train_config.eval_data_path,
        clap_tokenizer=model.clap_encoder.tokenizer,
        src_tokenizer=model.qwen_tokenizer,
        train_batch_size=train_config.generation_train_batch_size,
        eval_batch_size=train_config.generation_eval_batch_size,
        dataset_type="generation",
        max_asm_length=train_config.max_asm_length,
        max_name_length=train_config.max_name_length,
        num_workers=4
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 优化器 - 分组设置不同学习率
    # CLAP encoder: 较小学习率
    # Projection layer: 较大学习率
    # LoRA: 较大学习率
    
    optimizer_groups = [
        {
            "params": [p for n, p in model.clap_encoder.named_parameters() if p.requires_grad],
            "lr": train_config.generation_lr * 0.1,  # CLAP用较小学习率
            "name": "clap_encoder"
        },
        {
            "params": [p for n, p in model.projection.named_parameters() if p.requires_grad],
            "lr": train_config.generation_lr,
            "name": "projection"
        },
        {
            "params": [p for n, p in model.qwen.named_parameters() if p.requires_grad],
            "lr": train_config.generation_lr,
            "name": "qwen_lora"
        }
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=0.01
    )
    
    # 调度器
    num_training_steps = (len(train_loader) // train_config.gradient_accumulation_steps) * train_config.generation_epochs
    num_warmup_steps = int(num_training_steps * train_config.generation_warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 早停
    early_stopping = EarlyStopping(patience=5, mode="min")
    
    # 训练循环
    best_loss = float("inf")
    best_exact_match = 0
    
    for epoch in range(1, train_config.generation_epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{train_config.generation_epochs}")
        logger.info(f"{'='*50}")
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            train_config.gradient_accumulation_steps, logger
        )
        logger.info(f"Train loss: {train_loss:.4f}")
        # 验证
        val_loss, val_metrics = evaluate(model, val_loader, device, logger)
        logger.info(f"Validation loss: {val_loss:.4f}")
        logger.info(f"Validation metrics: {val_metrics}")
        # 保存最佳模型（基于loss）
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(output_dir, "best_model")
            model.save_pretrained(save_path)
            logger.info(f"Saved best model (loss) to {save_path}")
            
        # 也保存最佳exact match模型
        if val_metrics["exact_match"] > best_exact_match:
            best_exact_match = val_metrics["exact_match"]
            save_path = os.path.join(output_dir, "best_model_em")
            model.save_pretrained(save_path)
            logger.info(f"Saved best model (exact match) to {save_path}")
            
        # 定期保存checkpoint
        if epoch % 5 == 0:
            save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
            model.save_pretrained(save_path)
            
        # 早停检查
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
            
        print_gpu_memory()
        
    logger.info(f"\nTraining completed!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Best exact match: {best_exact_match:.4f}")
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Generation Training")
    parser.add_argument("--train_data_path", type=str, default="resources/sym-dataset/func_pairs_with_strings_train.csv", help="Path to CSV train data file")
    parser.add_argument("--eval_data_path", type=str, default="resources/sym-dataset/func_pairs_with_strings_eval.csv", help="Path to CSV eval data file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Eval batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--alignment_checkpoint", 
        type=str, 
        help="Path to alignment stage checkpoint (optional)"
    )
    
    args = parser.parse_args()
    main(args)
