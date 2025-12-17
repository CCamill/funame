"""
阶段2：生成训练

训练完整的ASM函数命名模型
"""
from datetime import datetime
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

from config import ModelConfig, TrainingConfig
from models import AsmNamingModel
from dataset import AsmFunctionDataset, create_dataloaders
from asm_function_naming_utils import ( 
    set_seed, setup_logger, save_config,
    AverageMeter, count_parameters,
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
    
    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}")
    
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
    
    for batch_idx, batch in enumerate(tqdm(val_loader, total=len(val_loader), desc="Evaluating")):
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


def main(model_config: ModelConfig, train_config: TrainingConfig):
    
    # 设置
    set_seed(train_config.seed)
    device = train_config.device
    output_dir = train_config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger("generation_training")
    save_config(train_config, output_dir, "training_config.json")
    save_config(model_config, output_dir, "model_config.json")
    
    # 创建模型
    model = AsmNamingModel(
        clap_model_name=model_config.clap_asm_model_name,
        llm_model_name=model_config.src_model_name,
        clap_hidden_size=model_config.clap_asm_hidden_size,
        llm_hidden_size=model_config.src_hidden_size,
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
        src_tokenizer=model.source_tokenizer,
        train_batch_size=train_config.generation_train_batch_size,
        eval_batch_size=train_config.generation_eval_batch_size,
        dataset_type="generation",
        max_asm_length=train_config.max_asm_length,
        max_name_length=train_config.max_name_length,
        num_workers=4,
        logger=logger
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
            "params": [p for n, p in model.source_model.named_parameters() if p.requires_grad],
            "lr": train_config.generation_lr,
            "name": "source_model"
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
    
    # 训练循环
    best_exact_match = 0
    
    for epoch in range(1, train_config.generation_epochs + 1):
        logger.info(f"{'-'*50}")
        logger.info(f"Epoch {epoch}/{train_config.generation_epochs}")
        
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
            
        
    logger.info(f"\nTraining completed!")
    logger.info(f"Best exact match: {best_exact_match:.4f}")
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Generation Training")
    parser.add_argument("--clap_model_name", type=str, default="hustcw/clap-asm", help="CLAP model name")
    parser.add_argument("--src_model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Source model name", choices=["Qwen/Qwen2.5-Coder-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"])
    parser.add_argument("--clap_hidden_size", type=int, default=768, help="CLAP hidden size")
    parser.add_argument("--src_hidden_size", type=int, default=2048, help="Source hidden size", choices=[2048, 4096])
    parser.add_argument("--projection_dim", type=int, default=768, help="Projection dimension")
    parser.add_argument("--use_4bit", type=bool, default=True, help="Use 4bit quantization")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--train_data_path", type=str, default="resources/sym-dataset/func_pairs_with_strings_train.csv", help="Path to CSV train data file")
    parser.add_argument("--eval_data_path", type=str, default="resources/sym-dataset/func_pairs_with_strings_eval.csv", help="Path to CSV eval data file")
    parser.add_argument("--output_dir", type=str, default=f"resources/asm-function-naming/generation-{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Eval batch size")
    parser.add_argument("--max_asm_length", type=int, default=1024, help="Max assembly code length")
    parser.add_argument("--max_src_length", type=int, default=512, help="Max source code length")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load checkpoint example: resources/asm-function-naming/generation-20251216_120407/Epoch_10")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--alignment_checkpoint", type=str, default="resources/asm-function-naming/alignment-20251216_120407/alignment/Epoch_10", help="Path to alignment stage checkpoint (optional), example: resources/asm-function-naming/alignment-20251216_120407/Epoch_10")
    args = parser.parse_args()
    if args.load_checkpoint:
        args.output_dir = os.path.dirname(args.load_checkpoint)

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
        generation_epochs=args.epochs,
        generation_train_batch_size=args.train_batch_size,
        generation_eval_batch_size=args.eval_batch_size,
        generation_lr=args.lr,
        generation_warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )
    
    main(model_config, train_config)
