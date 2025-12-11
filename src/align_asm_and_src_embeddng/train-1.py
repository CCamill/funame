"""
Stage 1 Training: Contrastive Learning
Align asm embeddings with src embeddings
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import get_scheduler
from tqdm import tqdm
import logging
from pathlib import Path
import json
from datetime import datetime

from config import ModelConfig, TrainingConfig, DataConfig
from model import AsmFunctionNamingModel
from data_utils import create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage1Trainer:
    """Trainer for Stage 1: Contrastive Learning"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Create directories
        os.makedirs(training_config.output_dir, exist_ok=True)
        os.makedirs(training_config.checkpoint_dir, exist_ok=True)
        os.makedirs(training_config.log_dir, exist_ok=True)
        
        # Initialize model
        logger.info("Initializing model for Stage 1...")
        self.model = AsmFunctionNamingModel(
            model_config=model_config,
            training_config=training_config,
            stage=1
        )
        self.model.to(training_config.device)
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.asm_tokenizer, self.src_tokenizer = \
            create_dataloaders(model_config, data_config, training_config, stage=1)
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.training_logs = []
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Only optimize CLAP-asm and adapter parameters
        optimizer_params = [
            {'params': self.model.clap_asm.parameters()},
            {'params': self.model.adapter.parameters()}
        ]
        
        self.optimizer = AdamW(
            optimizer_params,
            lr=self.training_config.stage1_learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Calculate total steps
        total_steps = len(self.train_loader) * self.training_config.stage1_epochs
        total_steps = total_steps // self.training_config.gradient_accumulation_steps
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Total training steps: {total_steps}")
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {
            'pos_cosine_sim': 0.0,
            'neg_cosine_sim': 0.0,
            'retrieval_accuracy': 0.0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.training_config.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            loss, metrics = self.model.forward_stage1(
                asm_input_ids=batch['asm_input_ids'],
                asm_attention_mask=batch['asm_attention_mask'],
                src_input_ids=batch['src_input_ids'],
                src_attention_mask=batch['src_attention_mask']
            )
            
            # Normalize loss for gradient accumulation
            loss = loss / self.training_config.gradient_accumulation_steps
            loss.backward()
            
            # Update metrics
            epoch_loss += metrics['contrastive_loss']
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            
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
                    'loss': f"{metrics['contrastive_loss']:.4f}",
                    'pos_sim': f"{metrics['pos_cosine_sim']:.4f}",
                    'neg_sim': f"{metrics['neg_cosine_sim']:.4f}",
                    'acc': f"{metrics['retrieval_accuracy']:.4f}"
                })
                
                # Logging
                if self.global_step % self.training_config.logging_steps == 0:
                    log_entry = {
                        'step': self.global_step,
                        'epoch': epoch,
                        'loss': metrics['contrastive_loss'],
                        'metrics': metrics,
                        'lr': self.scheduler.get_last_lr()[0]
                    }
                    self.training_logs.append(log_entry)
                
                # Save checkpoint
                if self.global_step % self.training_config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
                
                # Validation
                if self.global_step % self.training_config.eval_steps == 0:
                    val_metrics = self.validate()
                    logger.info(f"Validation at step {self.global_step}: {val_metrics}")
                    
                    # Save best model
                    if val_metrics['retrieval_accuracy'] > self.best_val_accuracy:
                        self.best_val_accuracy = val_metrics['retrieval_accuracy']
                        self.save_checkpoint("best_model")
                        logger.info(f"New best model! Accuracy: {self.best_val_accuracy:.4f}")
                    
                    self.model.train()
        
        # Average metrics
        num_batches = len(self.train_loader)
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_loss, epoch_metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = {
            'pos_cosine_sim': 0.0,
            'neg_cosine_sim': 0.0,
            'retrieval_accuracy': 0.0
        }
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = {k: v.to(self.training_config.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            loss, metrics = self.model.forward_stage1(
                asm_input_ids=batch['asm_input_ids'],
                asm_attention_mask=batch['asm_attention_mask'],
                src_input_ids=batch['src_input_ids'],
                src_attention_mask=batch['src_attention_mask']
            )
            
            val_loss += metrics['contrastive_loss']
            for key in val_metrics:
                val_metrics[key] += metrics[key]
        
        # Average
        num_batches = len(self.val_loader)
        val_loss /= num_batches
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        val_metrics['loss'] = val_loss
        return val_metrics
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.training_config.checkpoint_dir,
            f"stage1_{name}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model components
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
            'best_val_accuracy': self.best_val_accuracy,
        }
        with open(os.path.join(checkpoint_path, "training_state.json"), 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting Stage 1 training...")
        logger.info(f"Training for {self.training_config.stage1_epochs} epochs")
        logger.info(f"Batch size: {self.training_config.stage1_batch_size}")
        logger.info(f"Learning rate: {self.training_config.stage1_learning_rate}")
        
        for epoch in range(1, self.training_config.stage1_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.training_config.stage1_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            epoch_loss, epoch_metrics = self.train_epoch(epoch)
            
            logger.info(f"Epoch {epoch} completed:")
            logger.info(f"  Loss: {epoch_loss:.4f}")
            logger.info(f"  Positive cosine similarity: {epoch_metrics['pos_cosine_sim']:.4f}")
            logger.info(f"  Negative cosine similarity: {epoch_metrics['neg_cosine_sim']:.4f}")
            logger.info(f"  Retrieval accuracy: {epoch_metrics['retrieval_accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Validation metrics:")
            logger.info(f"  Loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Positive cosine similarity: {val_metrics['pos_cosine_sim']:.4f}")
            logger.info(f"  Negative cosine similarity: {val_metrics['neg_cosine_sim']:.4f}")
            logger.info(f"  Retrieval accuracy: {val_metrics['retrieval_accuracy']:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch}")
            
            # Check if we've achieved target performance
            if val_metrics['pos_cosine_sim'] > 0.95 and val_metrics['neg_cosine_sim'] < -0.5:
                logger.info("Target performance achieved! Training complete.")
                break
        
        # Save training logs
        log_path = os.path.join(
            self.training_config.log_dir,
            f"stage1_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(log_path, 'w') as f:
            json.dump(self.training_logs, f, indent=2)
        
        logger.info(f"Training logs saved to {log_path}")
        logger.info("Stage 1 training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")


def main():
    # Load configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # Create trainer
    trainer = Stage1Trainer(model_config, training_config, data_config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()