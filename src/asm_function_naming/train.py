#!/usr/bin/env python
"""
端到端训练脚本

自动执行两阶段训练：
1. 对比学习对齐（可选）
2. 生成训练
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd: list, description: str):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60 + "\n")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with return code {result.returncode}")
        return False
    return True


def main(args):
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           ASM Function Naming - End-to-End Training          ║
╠══════════════════════════════════════════════════════════════╣
║  Data: {args.data_path:<52} ║
║  Output: {args.output_dir:<50} ║
║  Skip Alignment: {str(args.skip_alignment):<42} ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    alignment_checkpoint = None
    
    # 阶段1：对比学习对齐
    if not args.skip_alignment:
        cmd = [
            sys.executable, "train_alignment.py",
            "--data_path", args.data_path,
            "--output_dir", args.output_dir,
            "--epochs", str(args.alignment_epochs),
            "--batch_size", str(args.alignment_batch_size),
            "--lr", str(args.alignment_lr)
        ]
        
        success = run_command(cmd, "Stage 1: Contrastive Learning Alignment")
        
        if not success:
            print("Stage 1 failed. Exiting.")
            return
            
        alignment_checkpoint = os.path.join(args.output_dir, "alignment", "best_alignment_model")
        print(f"\nAlignment checkpoint saved to: {alignment_checkpoint}")
    
    # 阶段2：生成训练
    cmd = [
        sys.executable, "train_generation.py",
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
        "--epochs", str(args.generation_epochs),
        "--batch_size", str(args.generation_batch_size),
        "--lr", str(args.generation_lr)
    ]
    
    if alignment_checkpoint:
        cmd.extend(["--alignment_checkpoint", alignment_checkpoint])
        
    success = run_command(cmd, "Stage 2: Generation Training")
    
    if not success:
        print("Stage 2 failed. Exiting.")
        return
        
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    Training Completed!                       ║
╠══════════════════════════════════════════════════════════════╣
║  Best model saved to: {os.path.join(args.output_dir, 'generation', 'best_model'):<36} ║
║                                                              ║
║  To run inference:                                           ║
║    python inference.py --checkpoint outputs/generation/best_model ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Training")
    
    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/sample_data.csv",
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory"
    )
    
    # Alignment stage
    parser.add_argument(
        "--skip_alignment",
        action="store_true",
        help="Skip alignment stage"
    )
    parser.add_argument(
        "--alignment_epochs",
        type=int,
        default=10,
        help="Number of alignment epochs"
    )
    parser.add_argument(
        "--alignment_batch_size",
        type=int,
        default=32,
        help="Alignment batch size"
    )
    parser.add_argument(
        "--alignment_lr",
        type=float,
        default=1e-4,
        help="Alignment learning rate"
    )
    
    # Generation stage
    parser.add_argument(
        "--generation_epochs",
        type=int,
        default=20,
        help="Number of generation epochs"
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=4,
        help="Generation batch size"
    )
    parser.add_argument(
        "--generation_lr",
        type=float,
        default=2e-5,
        help="Generation learning rate"
    )
    
    args = parser.parse_args()
    main(args)
