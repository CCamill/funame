"""
Model architecture for Assembly Function Naming
Combines CLAP-asm encoder with Qwen2.5-coder decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ProjectionAdapter(nn.Module):
    """
    Adapter to project CLAP-asm embeddings to Qwen hidden state space
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, hidden_size] - CLAP-asm embeddings
        Returns:
            [batch_size, output_dim] - Projected embeddings
        """
        return self.projection(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for aligning asm and src embeddings
    Uses InfoNCE loss with temperature scaling
    """
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(
        self, 
        asm_embeddings: torch.Tensor, 
        src_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            asm_embeddings: [batch_size, hidden_dim]
            src_embeddings: [batch_size, hidden_dim]
        Returns:
            loss: scalar
            metrics: dict with cosine similarities
        """
        # Normalize embeddings
        asm_embeddings = F.normalize(asm_embeddings, p=2, dim=1)
        src_embeddings = F.normalize(src_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        # [batch_size, batch_size]
        similarity_matrix = torch.matmul(asm_embeddings, src_embeddings.T) / self.temperature
        
        # Positive samples are on the diagonal
        batch_size = asm_embeddings.size(0)
        labels = torch.arange(batch_size, device=asm_embeddings.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        # Calculate metrics
        with torch.no_grad():
            # Positive similarities (diagonal)
            pos_sim = torch.diagonal(similarity_matrix).mean()
            
            # Negative similarities (off-diagonal)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=asm_embeddings.device)
            neg_sim = similarity_matrix[mask].mean()
            
            # Accuracy: top-1 retrieval
            pred = similarity_matrix.argmax(dim=1)
            accuracy = (pred == labels).float().mean()
        
        metrics = {
            'contrastive_loss': loss.item(),
            'pos_cosine_sim': pos_sim.item(),
            'neg_cosine_sim': neg_sim.item(),
            'retrieval_accuracy': accuracy.item()
        }
        
        return loss, metrics


class AsmFunctionNamingModel(nn.Module):
    """
    Main model that combines CLAP-asm and Qwen2.5-coder
    """
    def __init__(
        self, 
        model_config,
        training_config,
        stage: int = 1
    ):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.stage = stage
        
        # Load CLAP-asm model (encoder)
        logger.info(f"Loading CLAP-asm model from {model_config.clap_asm_model_name}")
        self.clap_asm = self._load_clap_asm_model()
        
        # Load Qwen2.5-coder model with 4-bit quantization
        logger.info(f"Loading Qwen2.5-coder model with {model_config.qwen_quantization} quantization")
        self.qwen_model, self.qwen_tokenizer = self._load_qwen_model()
        
        # Projection adapter
        self.adapter = ProjectionAdapter(
            input_dim=model_config.clap_asm_hidden_size,
            output_dim=model_config.adapter_hidden_size,
            dropout=model_config.adapter_dropout
        )
        
        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(
            temperature=training_config.stage1_temperature,
            margin=training_config.stage1_margin
        )
        
        # Set requires_grad based on stage
        self._configure_training_stage()
        
    def _load_clap_asm_model(self):
        """
        Load CLAP-asm model
        Replace this with actual CLAP-asm loading code
        """
        try:
            # Example: If CLAP-asm is a HuggingFace model
            model = AutoModel.from_pretrained(
                self.model_config.clap_asm_model_name,
                trust_remote_code=True
            )
            return model
        except Exception as e:
            logger.warning(f"Could not load CLAP-asm model: {e}")
            logger.info("Creating placeholder model for CLAP-asm")
            # Placeholder: Create a simple encoder
            return self._create_placeholder_clap_asm()
    
    def _create_placeholder_clap_asm(self):
        """Create a placeholder CLAP-asm model for testing"""
        class PlaceholderClapAsm(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=8,
                        dim_feedforward=hidden_size * 4,
                        batch_first=True
                    ),
                    num_layers=6
                )
                self.embedding = nn.Embedding(50000, hidden_size)
                self.pooler = nn.Linear(hidden_size, hidden_size)
                
            def forward(self, input_ids, attention_mask=None):
                embeds = self.embedding(input_ids)
                encoded = self.encoder(embeds)
                # Pool: mean of all tokens
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(-1).expand_as(encoded)
                    summed = (encoded * mask).sum(dim=1)
                    count = mask.sum(dim=1)
                    pooled = summed / count.clamp(min=1)
                else:
                    pooled = encoded.mean(dim=1)
                return self.pooler(pooled)
        
        return PlaceholderClapAsm(self.model_config.clap_asm_hidden_size)
    
    def _load_qwen_model(self):
        """Load Qwen2.5-coder with 4-bit quantization"""
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.qwen_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.qwen_model_name,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Apply LoRA for stage 2
        if self.stage == 2:
            lora_config = LoraConfig(
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                target_modules=self.model_config.lora_target_modules,
                lora_dropout=self.model_config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model = get_peft_model(model, lora_config)
            logger.info("LoRA applied to Qwen model")
            model.print_trainable_parameters()
        
        return model, tokenizer
    
    def _configure_training_stage(self):
        """Configure which parameters to train based on stage"""
        if self.stage == 1:
            # Stage 1: Train CLAP-asm and adapter
            logger.info("Stage 1: Training CLAP-asm and adapter")
            for param in self.qwen_model.parameters():
                param.requires_grad = False
            for param in self.clap_asm.parameters():
                param.requires_grad = True
            for param in self.adapter.parameters():
                param.requires_grad = True
                
        elif self.stage == 2:
            # Stage 2: Freeze CLAP-asm, adapter, train Qwen with LoRA
            logger.info("Stage 2: Training Qwen with LoRA")
            for param in self.clap_asm.parameters():
                param.requires_grad = False
            for param in self.adapter.parameters():
                param.requires_grad = False
            # LoRA parameters are already trainable
    
    def encode_asm(
        self, 
        asm_input_ids: torch.Tensor,
        asm_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode assembly code using CLAP-asm
        Args:
            asm_input_ids: [batch_size, seq_len]
            asm_attention_mask: [batch_size, seq_len]
        Returns:
            asm_embeddings: [batch_size, hidden_dim]
        """
        asm_embeddings = self.clap_asm(
            input_ids=asm_input_ids,
            attention_mask=asm_attention_mask
        )
        
        # If output is a dict or tuple, extract the embedding
        if isinstance(asm_embeddings, dict):
            asm_embeddings = asm_embeddings.get('pooler_output', asm_embeddings.get('last_hidden_state')[:, 0])
        elif isinstance(asm_embeddings, tuple):
            asm_embeddings = asm_embeddings[0][:, 0] if len(asm_embeddings[0].shape) == 3 else asm_embeddings[0]
        
        # Project to Qwen's hidden space
        projected_embeddings = self.adapter(asm_embeddings)
        
        return projected_embeddings
    
    def encode_src(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source code using Qwen2.5-coder (extract layer 16)
        Args:
            src_input_ids: [batch_size, seq_len]
            src_attention_mask: [batch_size, seq_len]
        Returns:
            src_embeddings: [batch_size, hidden_dim]
        """
        with torch.no_grad():  # Don't compute gradients for source encoding
            outputs = self.qwen_model(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract hidden state from target layer (layer 16)
            target_layer_output = outputs.hidden_states[self.model_config.qwen_target_layer]
            
            # Pool: use mean of all tokens or [CLS] token
            if src_attention_mask is not None:
                mask = src_attention_mask.unsqueeze(-1).expand_as(target_layer_output)
                summed = (target_layer_output * mask).sum(dim=1)
                count = mask.sum(dim=1)
                src_embeddings = summed / count.clamp(min=1)
            else:
                src_embeddings = target_layer_output.mean(dim=1)
        
        return src_embeddings
    
    def forward_stage1(
        self,
        asm_input_ids: torch.Tensor,
        asm_attention_mask: Optional[torch.Tensor],
        src_input_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage 1: Contrastive learning to align asm and src embeddings
        """
        asm_embeddings = self.encode_asm(asm_input_ids, asm_attention_mask)
        src_embeddings = self.encode_src(src_input_ids, src_attention_mask)
        
        loss, metrics = self.contrastive_loss(asm_embeddings, src_embeddings)
        
        return loss, metrics
    
    def forward_stage2(
        self,
        asm_input_ids: torch.Tensor,
        asm_attention_mask: Optional[torch.Tensor],
        function_name_input_ids: torch.Tensor,
        function_name_attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2: Generate function names from asm embeddings
        
        This is a simplified version. In practice, you might want to:
        1. Inject asm embeddings into layer 17 of Qwen
        2. Use the injected embeddings as context for generation
        
        For now, we'll use a prefix-based approach where asm embeddings
        are prepended to the input sequence.
        """
        # Encode asm to get embeddings
        asm_embeddings = self.encode_asm(asm_input_ids, asm_attention_mask)
        
        # Get input embeddings for function names
        inputs_embeds = self.qwen_model.get_input_embeddings()(function_name_input_ids)
        
        # Prepend asm embeddings as a special "prefix" token
        # [batch_size, 1, hidden_dim]
        asm_embeddings = asm_embeddings.unsqueeze(1)
        
        # Concatenate: [batch_size, 1 + seq_len, hidden_dim]
        combined_embeds = torch.cat([asm_embeddings, inputs_embeds], dim=1)
        
        # Adjust attention mask
        batch_size = asm_embeddings.size(0)
        prefix_mask = torch.ones(batch_size, 1, device=asm_embeddings.device, dtype=function_name_attention_mask.dtype)
        combined_attention_mask = torch.cat([prefix_mask, function_name_attention_mask], dim=1)
        
        # Forward pass
        outputs = self.qwen_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=function_name_input_ids,  # Shifted inside the model
            return_dict=True
        )
        
        return outputs.loss, outputs.logits
    
    def generate(
        self,
        asm_input_ids: torch.Tensor,
        asm_attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 32,
        num_beams: int = 4
    ) -> torch.Tensor:
        """
        Generate function names from assembly code
        """
        self.eval()
        with torch.no_grad():
            # Encode asm
            asm_embeddings = self.encode_asm(asm_input_ids, asm_attention_mask)
            
            # Create a prompt embedding
            batch_size = asm_embeddings.size(0)
            asm_embeddings = asm_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Create initial input (just the embedding)
            # We'll need to handle this carefully based on Qwen's generation API
            
            # For simplicity, use a text prompt approach
            prompt = "Function name:"
            prompt_ids = self.qwen_tokenizer(
                [prompt] * batch_size,
                return_tensors="pt",
                padding=True
            ).input_ids.to(asm_embeddings.device)
            
            # Get prompt embeddings
            prompt_embeds = self.qwen_model.get_input_embeddings()(prompt_ids)
            
            # Combine
            combined_embeds = torch.cat([asm_embeddings, prompt_embeds], dim=1)
            
            # Generate
            # Note: This is simplified. You may need to implement custom generation logic
            # to properly inject the embeddings at layer 17
            
            outputs = self.qwen_model.generate(
                inputs_embeds=combined_embeds,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.qwen_tokenizer.pad_token_id,
                eos_token_id=self.qwen_tokenizer.eos_token_id
            )
            
            return outputs