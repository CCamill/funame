"""
Inference script for generating function names from assembly code
"""
import torch
import argparse
import json
from typing import List, Dict
from pathlib import Path
import logging

from config import ModelConfig, TrainingConfig, DataConfig
from model import AsmFunctionNamingModel
from data_utils import create_tokenizers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsmFunctionNamePredictor:
    """Predictor for generating function names from assembly code"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        checkpoint_path: str
    ):
        self.model_config = model_config
        self.training_config = training_config
        
        logger.info("Loading model...")
        
        # Initialize model (stage 2 for generation)
        self.model = AsmFunctionNamingModel(
            model_config=model_config,
            training_config=training_config,
            stage=2
        )
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Move to device and set to eval mode
        self.model.to(training_config.device)
        self.model.eval()
        
        # Get tokenizers
        self.asm_tokenizer, self.src_tokenizer = create_tokenizers(model_config)
        
        logger.info("Model loaded successfully!")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load CLAP-asm
        clap_asm_path = Path(checkpoint_path) / "clap_asm.pt"
        if clap_asm_path.exists():
            self.model.clap_asm.load_state_dict(
                torch.load(clap_asm_path, map_location='cpu')
            )
            logger.info("Loaded CLAP-asm weights")
        
        # Load adapter
        adapter_path = Path(checkpoint_path) / "adapter.pt"
        if adapter_path.exists():
            self.model.adapter.load_state_dict(
                torch.load(adapter_path, map_location='cpu')
            )
            logger.info("Loaded adapter weights")
        
        # Load LoRA weights
        try:
            from peft import PeftModel
            self.model.qwen_model = PeftModel.from_pretrained(
                self.model.qwen_model,
                checkpoint_path
            )
            logger.info("Loaded LoRA weights")
        except Exception as e:
            logger.warning(f"Could not load LoRA weights: {e}")
    
    @torch.no_grad()
    def predict(
        self,
        asm_code: str,
        max_length: int = 32,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        Predict function name from assembly code
        
        Args:
            asm_code: Assembly code string
            max_length: Maximum length of generated name
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Predicted function name
        """
        # Tokenize assembly code
        asm_inputs = self.asm_tokenizer(
            asm_code,
            max_length=2048,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        asm_inputs = {k: v.to(self.training_config.device) for k, v in asm_inputs.items()}
        
        # Generate
        generated_ids = self.model.generate(
            asm_input_ids=asm_inputs['input_ids'],
            asm_attention_mask=asm_inputs['attention_mask'],
            max_length=max_length,
            num_beams=num_beams
        )
        
        # Decode
        predicted_name = self.src_tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        ).strip()
        
        # Clean up the output (remove prompt if present)
        if "Function name:" in predicted_name:
            predicted_name = predicted_name.split("Function name:")[-1].strip()
        
        return predicted_name
    
    def predict_batch(
        self,
        asm_codes: List[str],
        max_length: int = 32,
        num_beams: int = 4,
        batch_size: int = 8
    ) -> List[str]:
        """
        Predict function names for a batch of assembly codes
        
        Args:
            asm_codes: List of assembly code strings
            max_length: Maximum length of generated names
            num_beams: Number of beams for beam search
            batch_size: Batch size for processing
            
        Returns:
            List of predicted function names
        """
        predictions = []
        
        for i in range(0, len(asm_codes), batch_size):
            batch = asm_codes[i:i+batch_size]
            
            # Tokenize
            asm_inputs = self.asm_tokenizer(
                batch,
                max_length=2048,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            asm_inputs = {k: v.to(self.training_config.device) for k, v in asm_inputs.items()}
            
            # Generate
            generated_ids = self.model.generate(
                asm_input_ids=asm_inputs['input_ids'],
                asm_attention_mask=asm_inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams
            )
            
            # Decode
            for ids in generated_ids:
                predicted_name = self.src_tokenizer.decode(
                    ids,
                    skip_special_tokens=True
                ).strip()
                
                # Clean up
                if "Function name:" in predicted_name:
                    predicted_name = predicted_name.split("Function name:")[-1].strip()
                
                predictions.append(predicted_name)
        
        return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate function names from assembly code")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Single assembly code string or path to JSON file'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='Path to JSON file with assembly codes'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Output file for predictions'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=32,
        help='Maximum length of generated names'
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=4,
        help='Number of beams for beam search'
    )
    args = parser.parse_args()
    
    # Load configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # Create predictor
    predictor = AsmFunctionNamePredictor(
        model_config=model_config,
        training_config=training_config,
        checkpoint_path=args.checkpoint
    )
    
    # Single prediction
    if args.input:
        if args.input.endswith('.json'):
            with open(args.input, 'r') as f:
                data = json.load(f)
            asm_code = data['asm_code'] if isinstance(data, dict) else data[0]['asm_code']
        else:
            asm_code = args.input
        
        logger.info("Generating function name...")
        predicted_name = predictor.predict(
            asm_code=asm_code,
            max_length=args.max_length,
            num_beams=args.num_beams
        )
        
        logger.info(f"Predicted function name: {predicted_name}")
        
        # Save result
        result = {
            'asm_code': asm_code,
            'predicted_name': predicted_name
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to {args.output}")
    
    # Batch prediction
    elif args.input_file:
        logger.info(f"Loading data from {args.input_file}")
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        asm_codes = [item['asm_code'] for item in data]
        
        logger.info(f"Generating function names for {len(asm_codes)} samples...")
        predictions = predictor.predict_batch(
            asm_codes=asm_codes,
            max_length=args.max_length,
            num_beams=args.num_beams
        )
        
        # Combine with original data
        results = []
        for item, pred in zip(data, predictions):
            result = {
                'asm_code': item['asm_code'],
                'predicted_name': pred
            }
            if 'function_name' in item:
                result['true_name'] = item['function_name']
            if 'src_code' in item:
                result['src_code'] = item['src_code']
            results.append(result)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")
        
        # Calculate accuracy if true names available
        if 'function_name' in data[0]:
            correct = sum(1 for r in results if r['predicted_name'] == r['true_name'])
            accuracy = correct / len(results) * 100
            logger.info(f"Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")
    
    else:
        logger.error("Please provide either --input or --input_file")
        return


if __name__ == "__main__":
    main()