"""
推理脚本

使用训练好的模型预测汇编函数的函数名
"""
import os
import sys
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_model_config, get_inference_config
from models import AsmNamingModel
from utils import set_seed, compute_metrics


def load_model(checkpoint_path: str, device: str = "cuda") -> AsmNamingModel:
    """
    加载训练好的模型
    """
    model_config = get_model_config()
    
    print(f"Loading model from {checkpoint_path}...")
    
    # 创建模型
    model = AsmNamingModel(
        clap_model_name=model_config.clap_asm_model_name,
        qwen_model_name=model_config.qwen_model_name,
        clap_hidden_size=model_config.clap_asm_hidden_size,
        qwen_hidden_size=model_config.qwen_hidden_size,
        num_prefix_tokens=model_config.num_prefix_tokens,
        projection_type="mlp",
        use_4bit=model_config.use_4bit,
        use_lora=False,  # 先不加载LoRA
        device=device
    )
    
    # 加载权重
    # 1. CLAP encoder
    clap_path = os.path.join(checkpoint_path, "clap_encoder.pt")
    if os.path.exists(clap_path):
        model.clap_encoder.load_state_dict(
            torch.load(clap_path, map_location=device)
        )
        print("Loaded CLAP encoder weights")
        
    # 2. Projection
    proj_path = os.path.join(checkpoint_path, "projection.pt")
    if os.path.exists(proj_path):
        model.projection.load_state_dict(
            torch.load(proj_path, map_location=device)
        )
        print("Loaded projection weights")
        
    # 3. Qwen LoRA
    lora_path = os.path.join(checkpoint_path, "qwen_lora")
    if os.path.exists(lora_path):
        from peft import PeftModel
        model.qwen = PeftModel.from_pretrained(
            model.qwen,
            lora_path
        )
        print("Loaded LoRA weights")
        
    model.eval()
    return model


def predict_single(
    model: AsmNamingModel,
    asm_code: str,
    max_new_tokens: int = 64,
    temperature: float = 0.1,
    do_sample: bool = False,
    num_beams: int = 1
) -> str:
    """
    预测单个汇编函数的函数名
    """
    return model.generate(
        asm_code=asm_code,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        num_beams=num_beams
    )


def predict_batch(
    model: AsmNamingModel,
    asm_codes: List[str],
    max_new_tokens: int = 64,
    temperature: float = 0.1,
    do_sample: bool = False,
    num_beams: int = 1,
    show_progress: bool = True
) -> List[str]:
    """
    批量预测函数名
    """
    predictions = []
    
    iterator = tqdm(asm_codes, desc="Predicting") if show_progress else asm_codes
    
    for asm_code in iterator:
        pred = predict_single(
            model, asm_code, max_new_tokens, temperature, do_sample, num_beams
        )
        predictions.append(pred)
        
    return predictions


def predict_from_csv(
    model: AsmNamingModel,
    csv_path: str,
    output_path: str,
    asm_column: str = "asm_code",
    name_column: Optional[str] = "func_name",
    max_new_tokens: int = 64
):
    """
    从CSV文件批量预测并保存结果
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} samples")
    
    # 预测
    asm_codes = df[asm_column].tolist()
    predictions = predict_batch(model, asm_codes, max_new_tokens=max_new_tokens)
    
    # 添加预测结果
    df["predicted_name"] = predictions
    
    # 如果有真实标签，计算指标
    if name_column and name_column in df.columns:
        references = df[name_column].tolist()
        metrics = compute_metrics(predictions, references)
        
        print("\nEvaluation Metrics:")
        print(f"  Exact Match: {metrics['exact_match']:.4f}")
        print(f"  Prefix Match: {metrics['prefix_match']:.4f}")
        print(f"  Avg Edit Distance: {metrics['avg_edit_distance']:.4f}")
        print(f"  Avg Char F1: {metrics['avg_char_f1']:.4f}")
        
        # 添加匹配标记
        df["exact_match"] = [
            p.lower() == r.lower() 
            for p, r in zip(predictions, references)
        ]
        
    # 保存结果
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    return df


def interactive_mode(model: AsmNamingModel):
    """
    交互式预测模式
    """
    print("\n" + "="*50)
    print("Interactive Mode")
    print("Enter assembly code (end with empty line)")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        print("Enter assembly code:")
        lines = []
        while True:
            line = input()
            if line.lower() == "quit":
                print("Goodbye!")
                return
            if line == "":
                break
            lines.append(line)
            
        if not lines:
            continue
            
        asm_code = "\n".join(lines)
        
        print("\nPredicting...")
        pred_name = predict_single(model, asm_code)
        print(f"\nPredicted function name: {pred_name}")
        print("-" * 50 + "\n")


def main(args):
    set_seed(42)
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    if args.mode == "single":
        # 单个预测
        if args.asm_code:
            pred = predict_single(
                model, 
                args.asm_code,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_beams=args.num_beams
            )
            print(f"\nPredicted function name: {pred}")
        else:
            print("Error: --asm_code is required for single mode")
            
    elif args.mode == "file":
        # 从文件预测
        if args.input_file and args.output_file:
            predict_from_csv(
                model,
                args.input_file,
                args.output_file,
                asm_column=args.asm_column,
                name_column=args.name_column,
                max_new_tokens=args.max_new_tokens
            )
        else:
            print("Error: --input_file and --output_file are required for file mode")
            
    elif args.mode == "interactive":
        # 交互式模式
        interactive_mode(model)
        
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASM Function Naming Inference")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "file", "interactive"],
        default="interactive",
        help="Inference mode"
    )
    
    # Single mode arguments
    parser.add_argument(
        "--asm_code",
        type=str,
        help="Assembly code string (for single mode)"
    )
    
    # File mode arguments
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input CSV file path (for file mode)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output CSV file path (for file mode)"
    )
    parser.add_argument(
        "--asm_column",
        type=str,
        default="asm_code",
        help="Column name for assembly code in CSV"
    )
    parser.add_argument(
        "--name_column",
        type=str,
        default="func_name",
        help="Column name for function name in CSV (optional, for evaluation)"
    )
    
    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search"
    )
    
    args = parser.parse_args()
    main(args)
