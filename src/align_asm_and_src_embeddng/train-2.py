"""
Stage 2 Training: Function Name Generation
Use aligned asm embeddings to generate function names with Qwen
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

from config import ModelConfig, TrainingConfig, DataConfig
from model import AsmFunctionNamingModel
from data_utils import create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage2Trainer:
    """Trainer for Stage 2: Function Name Generation"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        stage1_checkpoint_path: str = None
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Create directories
        os.makedirs(training_config.output_dir, exist_ok=True)
        os.makedirs(training_config.checkpoint_dir, exist_ok=True)
        os.makedirs(training_config.log_dir, exist_ok=True)
        
        # Initialize model
        logger.info("Initializing model for Stage 2...")
        self.model = AsmFunctionNamingModel(
            model_config=model_config,
            training_config=training_config,
            stage=2
        )
        
        # Load Stage 1 checkpoint
        if stage1_checkpoint_path:
            self.load_stage1_checkpoint(stage1_checkpoint_path)
        
        self.model.to(training_config.device)
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.asm_tokenizer, self.src_tokenizer = \
            create_dataloaders(model_config, data_config, training_config, stage=2)
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_logs = []
        
    def load_stage1_checkpoint(self, checkpoint_path: str):
        """Load CLAP-asm and adapter weights from Stage 1"""
        logger.info(f"Loading Stage 1 checkpoint from {checkpoint_path}")
        
        clap_asm_path = os.path.join(checkpoint_path, "clap_asm.pt")
        adapter_path = os.path.join(checkpoint_path, "adapter.pt")
        
        if os.path.exists(clap_asm_path):
            self.model.clap_asm.load_state_dict(
                torch.load(clap_asm_path, map_location='cpu')
            )
            logger.info("Loaded CLAP-asm weights")
        
        if os.path.exists(adapter_path):
            self.model.adapter.load_state_dict(
                torch.load(adapter_path, map_location='cpu')
            )
            logger.info("Loaded adapter weights")
        
        # Freeze CLAP-asm and adapter
        for param in self.model.clap_asm.parameters():
            param.requires_grad = False
        for param in self.model.adapter.parameters():
            param.requires_grad = False
        
        logger.info("Froze CLAP-asm and adapter parameters")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler - only for LoRA parameters"""
        # Only optimize LoRA parameters (already configured in model)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.training_config.stage2_learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Calculate total steps
        total_steps = len(self.train_loader) * self.training_config.stage2_epochs
        total_steps = total_steps // self.training_config.gradient_accumulation_steps
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Total training steps: {total_steps}")
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.training_config.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            loss, logits = self.model.forward_stage2(
                asm_input_ids=batch['asm_input_ids'],
                asm_attention_mask=batch['asm_attention_mask'],
                function_name_input_ids=batch['function_name_input_ids'],
                function_name_attention_mask=batch['function_name_attention_mask']
            )
            
            # Normalize loss for gradient accumulation
            loss = loss / self.training_config.gradient_accumulation_steps
            loss.backward()
            
            # Update metrics
            epoch_loss += loss.item() * self.training_config.gradient_accumulation_steps
            
            # Gradient accumulation
            if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * self.training_config.gradient_accumulation_steps:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Logging
                if self.global_step % self.training_config.logging_steps == 0:
                    log_entry = {
                        'step': self.global_step,
                        'epoch': epoch,
                        'loss': loss.item() * self.training_config.gradient_accumulation_steps,
                        'lr': self.scheduler.get_last_lr()[0]
                    }
                    self.training_logs.append(log_entry)
                
                # Save checkpoint
                if self.global_step % self.training_config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
                
                # Validation
                if self.global_step % self.training_config.eval_steps == 0:
                    val_loss, val_samples = self.validate()
                    logger.info(f"\nValidation at step {self.global_step}:")
                    logger.info(f"  Loss: {val_loss:.4f}")
                    logger.info("  Sample predictions:")
                    for i, sample in enumerate(val_samples[:3]):
                        logger.info(f"    {i+1}. True: {sample['true']} | Pred: {sample['pred']}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model")
                        logger.info(f"New best model! Loss: {self.best_val_loss:.4f}")
                    
                    self.model.train()
        
        # Average loss
        epoch_loss /= len(self.train_loader)
        
        return epoch_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        val_loss = 0.0
        sample_predictions = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            batch_cpu = {k: v for k, v in batch.items()}  # Keep original for decoding
            batch = {k: v.to(self.training_config.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Calculate loss
            loss, logits = self.model.forward_stage2(
                asm_input_ids=batch['asm_input_ids'],
                asm_attention_mask=batch['asm_attention_mask'],
                function_name_input_ids=batch['function_name_input_ids'],
                function_name_attention_mask=batch['function_name_attention_mask']
            )
            
            val_loss += loss.item()
            
            # Generate predictions for a few samples
            if len(sample_predictions) < 20:
                try:
                    generated_ids = self.model.generate(
                        asm_input_ids=batch['asm_input_ids'][:2],  # First 2 samples
                        asm_attention_mask=batch['asm_attention_mask'][:2],
                        max_length=self.training_config.stage2_max_length,
                        num_beams=4
                    )
                    
                    for i in range(min(2, len(batch_cpu['function_names']))):
                        pred_text = self.src_tokenizer.decode(
                            generated_ids[i],
                            skip_special_tokens=True
                        )
                        sample_predictions.append({
                            'true': batch_cpu['function_names'][i],
                            'pred': pred_text
                        })
                except Exception as e:
                    logger.warning(f"Error during generation: {e}")
        
        # Average loss
        val_loss /= len(self.val_loader)
        
        return val_loss, sample_predictions
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.training_config.checkpoint_dir,
            f"stage2_{name}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save LoRA weights
        if hasattr(self.model.qwen_model, 'save_pretrained'):
            self.model.qwen_model.save_pretrained(checkpoint_path)
            logger.info(f"Saved LoRA weights to {checkpoint_path}")
        
        # Also save full model components for inference
        torch.save(
            self.model.clap_asm.state_dict(),
            os.path.join(checkpoint_path, "clap_asm.pt")
        )
        torch.save(
            self.model.adapter.state_dict(),
            os.path.join(checkpoint_path, "adapter.pt")
        )
        
        # Save optimizer and scheduler
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_path, "optimizer.pt")
        )
        torch.save(
            self.scheduler.state_dict(),
            os.path.join(checkpoint_path, "scheduler.pt")
        )
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }
        with open(os.path.join(checkpoint_path, "training_state.json"), 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting Stage 2 training...")
        logger.info(f"Training for {self.training_config.stage2_epochs} epochs")
        logger.info(f"Batch size: {self.training_config.stage2_batch_size}")
        logger.info(f"Learning rate: {self.training_config.stage2_learning_rate}")
        
        for epoch in range(1, self.training_config.stage2_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.training_config.stage2_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            epoch_loss = self.train_epoch(epoch)
            
            logger.info(f"Epoch {epoch} completed:")
            logger.info(f"  Loss: {epoch_loss:.4f}")
            
            # Validate
            val_loss, val_samples = self.validate()
            logger.info(f"Validation metrics:")
            logger.info(f"  Loss: {val_loss:.4f}")
            logger.info("  Sample predictions:")
            for i, sample in enumerate(val_samples[:5]):
                logger.info(f"    {i+1}. True: {sample['true']}")
                logger.info(f"       Pred: {sample['pred']}")
            
            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch}")
        
        # Save training logs
        log_path = os.path.join(
            self.training_config.log_dir,
            f"stage2_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(log_path, 'w') as f:
            json.dump(self.training_logs, f, indent=2)
        
        logger.info(f"Training logs saved to {log_path}")
        logger.info("Stage 2 training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stage1_checkpoint',
        type=str,
        default='./checkpoints/stage1_best_model',
        help='Path to Stage 1 checkpoint'
    )
    args = parser.parse_args()
    
    # Load configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # Create trainer
    trainer = Stage2Trainer(
        model_config,
        training_config,
        data_config,
        stage1_checkpoint_path=args.stage1_checkpoint
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()