"""
Data processing and dataset classes
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import random
import logging

logger = logging.getLogger(__name__)


class AsmFunctionDataset(Dataset):
    """
    Dataset for assembly function naming task
    Expected data format:
    {
        "asm_code": "assembly code string",
        "src_code": "source code string",
        "function_name": "function_name"
    }
    """
    def __init__(
        self,
        data_path: str,
        asm_tokenizer,  # Tokenizer for CLAP-asm
        src_tokenizer,  # Tokenizer for Qwen
        data_config,
        stage: int = 1
    ):
        self.data_config = data_config
        self.asm_tokenizer = asm_tokenizer
        self.src_tokenizer = src_tokenizer
        self.stage = stage
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tokenize assembly code
        asm_encoded = self.asm_tokenizer(
            sample['asm_code'],
            max_length=self.data_config.max_asm_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'asm_input_ids': asm_encoded['input_ids'].squeeze(0),
            'asm_attention_mask': asm_encoded['attention_mask'].squeeze(0),
        }
        
        if self.stage == 1:
            # Stage 1: Need source code for contrastive learning
            src_encoded = self.src_tokenizer(
                sample['src_code'],
                max_length=self.data_config.max_src_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            result.update({
                'src_input_ids': src_encoded['input_ids'].squeeze(0),
                'src_attention_mask': src_encoded['attention_mask'].squeeze(0),
            })
        
        elif self.stage == 2:
            # Stage 2: Need function names for generation
            function_name = sample['function_name']
            
            # Create input-output pair for causal LM
            # Input: "<asm_embedding> Generate function name:"
            # Output: "function_name"
            fn_encoded = self.src_tokenizer(
                function_name,
                max_length=self.data_config.max_src_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            result.update({
                'function_name_input_ids': fn_encoded['input_ids'].squeeze(0),
                'function_name_attention_mask': fn_encoded['attention_mask'].squeeze(0),
                'function_name': function_name  # Keep original for evaluation
            })
        
        return result


class ContrastiveDataCollator:
    """
    Custom collator for stage 1 that creates negative samples
    """
    def __init__(self, num_negatives: int = 4):
        self.num_negatives = num_negatives
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch and create negative samples
        For each sample, we'll use other samples in the batch as negatives
        """
        # Stack tensors
        asm_input_ids = torch.stack([item['asm_input_ids'] for item in batch])
        asm_attention_mask = torch.stack([item['asm_attention_mask'] for item in batch])
        src_input_ids = torch.stack([item['src_input_ids'] for item in batch])
        src_attention_mask = torch.stack([item['src_attention_mask'] for item in batch])
        
        return {
            'asm_input_ids': asm_input_ids,
            'asm_attention_mask': asm_attention_mask,
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attention_mask,
        }


class GenerationDataCollator:
    """
    Custom collator for stage 2 (generation)
    """
    def __call__(self, batch: List[Dict]) -> Dict:
        asm_input_ids = torch.stack([item['asm_input_ids'] for item in batch])
        asm_attention_mask = torch.stack([item['asm_attention_mask'] for item in batch])
        function_name_input_ids = torch.stack([item['function_name_input_ids'] for item in batch])
        function_name_attention_mask = torch.stack([item['function_name_attention_mask'] for item in batch])
        function_names = [item['function_name'] for item in batch]
        
        return {
            'asm_input_ids': asm_input_ids,
            'asm_attention_mask': asm_attention_mask,
            'function_name_input_ids': function_name_input_ids,
            'function_name_attention_mask': function_name_attention_mask,
            'function_names': function_names
        }


def create_tokenizers(model_config):
    """
    Create tokenizers for CLAP-asm and Qwen
    """
    # CLAP-asm tokenizer
    try:
        asm_tokenizer = AutoTokenizer.from_pretrained(
            model_config.clap_asm_model_name,
            trust_remote_code=True
        )
    except Exception as e:
        logger.warning(f"Could not load CLAP-asm tokenizer: {e}")
        logger.info("Using Qwen tokenizer as placeholder for asm")
        asm_tokenizer = AutoTokenizer.from_pretrained(
            model_config.qwen_model_name,
            trust_remote_code=True
        )
    
    # Qwen tokenizer
    src_tokenizer = AutoTokenizer.from_pretrained(
        model_config.qwen_model_name,
        trust_remote_code=True
    )
    
    if asm_tokenizer.pad_token is None:
        asm_tokenizer.pad_token = asm_tokenizer.eos_token
    if src_tokenizer.pad_token is None:
        src_tokenizer.pad_token = src_tokenizer.eos_token
    
    return asm_tokenizer, src_tokenizer


def create_dataloaders(
    model_config,
    data_config,
    training_config,
    stage: int = 1
):
    """
    Create train and validation dataloaders
    """
    asm_tokenizer, src_tokenizer = create_tokenizers(model_config)
    
    # Create datasets
    train_dataset = AsmFunctionDataset(
        data_path=data_config.train_data_path,
        asm_tokenizer=asm_tokenizer,
        src_tokenizer=src_tokenizer,
        data_config=data_config,
        stage=stage
    )
    
    val_dataset = AsmFunctionDataset(
        data_path=data_config.val_data_path,
        asm_tokenizer=asm_tokenizer,
        src_tokenizer=src_tokenizer,
        data_config=data_config,
        stage=stage
    )
    
    # Select collator based on stage
    if stage == 1:
        collator = ContrastiveDataCollator(num_negatives=data_config.num_negatives)
        batch_size = training_config.stage1_batch_size
    else:
        collator = GenerationDataCollator()
        batch_size = training_config.stage2_batch_size
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, asm_tokenizer, src_tokenizer


def create_sample_data(output_path: str, num_samples: int = 1000):
    """
    Create sample data for testing
    This generates synthetic data - replace with your actual data
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    samples = []
    function_names = [
        "calculate_sum", "process_data", "init_system", "handle_request",
        "parse_input", "validate_user", "encrypt_data", "decode_message",
        "allocate_memory", "release_resource", "update_config", "fetch_result"
    ]
    
    for i in range(num_samples):
        # Generate synthetic assembly code
        asm_code = f"""
        push rbp
        mov rbp, rsp
        sub rsp, 0x{i:02x}
        mov DWORD PTR [rbp-0x4], edi
        mov DWORD PTR [rbp-0x8], esi
        mov eax, DWORD PTR [rbp-0x4]
        add eax, DWORD PTR [rbp-0x8]
        leave
        ret
        """
        
        # Generate synthetic source code
        src_code = f"""
        int function_{i}(int a, int b) {{
            int result = a + b;
            return result;
        }}
        """
        
        # Random function name
        function_name = random.choice(function_names) + f"_{i % 100}"
        
        samples.append({
            "asm_code": asm_code.strip(),
            "src_code": src_code.strip(),
            "function_name": function_name
        })
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {num_samples} samples at {output_path}")


if __name__ == "__main__":
    # Create sample datasets
    logging.basicConfig(level=logging.INFO)
    
    create_sample_data("./data/train.json", num_samples=800)
    create_sample_data("./data/val.json", num_samples=100)
    create_sample_data("./data/test.json", num_samples=100)
    
    print("Sample data created successfully!")