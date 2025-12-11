#!/usr/bin/env python
"""
快速测试脚本

验证所有模块是否可以正常导入和基本功能是否正常
"""
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """测试导入"""
    print("Testing imports...")
    
    try:
        from config import get_model_config, get_training_config, get_inference_config
        print("  ✓ config")
    except Exception as e:
        print(f"  ✗ config: {e}")
        return False
        
    try:
        from models import CLAPAsmEncoder, AsmProjection, AsmNamingModel
        print("  ✓ models")
    except Exception as e:
        print(f"  ✗ models: {e}")
        return False
        
    try:
        from dataset import AsmFunctionDataset, AsmAlignmentDataset
        print("  ✓ dataset")
    except Exception as e:
        print(f"  ✗ dataset: {e}")
        return False
        
    try:
        from utils import set_seed, compute_metrics, AverageMeter
        print("  ✓ utils")
    except Exception as e:
        print(f"  ✗ utils: {e}")
        return False
        
    return True


def test_config():
    """测试配置"""
    print("\nTesting config...")
    
    from config import get_model_config, get_training_config
    
    model_config = get_model_config()
    train_config = get_training_config()
    
    print(f"  Model: {model_config.qwen_model_name}")
    print(f"  CLAP: {model_config.clap_asm_model_name}")
    print(f"  Prefix tokens: {model_config.num_prefix_tokens}")
    print(f"  4bit quantization: {model_config.use_4bit}")
    
    return True


def test_data():
    """测试数据加载"""
    print("\nTesting data loading...")
    
    import pandas as pd
    
    data_path = "data/sample_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"  ✓ Loaded {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        
        # 显示一个样本
        if len(df) > 0:
            print(f"\n  Sample 0:")
            print(f"    func_name: {df.iloc[0]['func_name']}")
            print(f"    src_code: {df.iloc[0]['src_code'][:50]}...")
            print(f"    asm_code: {df.iloc[0]['asm_code'][:50]}...")
    else:
        print(f"  ✗ Data file not found: {data_path}")
        return False
        
    return True


def test_projection():
    """测试投影层"""
    print("\nTesting projection layer...")
    
    import torch
    from models.projection import AsmProjection, CrossAttentionProjection, GatedProjection
    
    batch_size = 2
    clap_hidden = 768
    llm_hidden = 2048
    num_prefix = 32
    
    # 测试MLP投影
    proj = AsmProjection(
        clap_hidden_size=clap_hidden,
        llm_hidden_size=llm_hidden,
        num_prefix_tokens=num_prefix
    )
    
    x = torch.randn(batch_size, clap_hidden)
    y = proj(x)
    
    expected_shape = (batch_size, num_prefix, llm_hidden)
    if y.shape == expected_shape:
        print(f"  ✓ MLP Projection: {x.shape} -> {y.shape}")
    else:
        print(f"  ✗ MLP Projection: expected {expected_shape}, got {y.shape}")
        return False
        
    # 测试CrossAttention投影
    proj2 = CrossAttentionProjection(
        clap_hidden_size=clap_hidden,
        llm_hidden_size=llm_hidden,
        num_prefix_tokens=num_prefix
    )
    y2 = proj2(x)
    
    if y2.shape == expected_shape:
        print(f"  ✓ CrossAttention Projection: {x.shape} -> {y2.shape}")
    else:
        print(f"  ✗ CrossAttention Projection: expected {expected_shape}, got {y2.shape}")
        return False
        
    return True


def test_metrics():
    """测试评估指标"""
    print("\nTesting metrics...")
    
    from utils import compute_metrics
    
    predictions = ["add", "multiply", "max_value", "swap_items"]
    references = ["add", "multiply", "max", "swap"]
    
    metrics = compute_metrics(predictions, references)
    
    print(f"  Exact Match: {metrics['exact_match']:.4f}")
    print(f"  Prefix Match: {metrics['prefix_match']:.4f}")
    print(f"  Avg Edit Distance: {metrics['avg_edit_distance']:.4f}")
    print(f"  Avg Char F1: {metrics['avg_char_f1']:.4f}")
    
    # 验证exact match应该是0.5（add和multiply完全匹配）
    if metrics['exact_match'] == 0.5:
        print("  ✓ Metrics calculation correct")
    else:
        print("  ✗ Metrics calculation incorrect")
        return False
        
    return True


def test_gpu():
    """测试GPU"""
    print("\nTesting GPU...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  ⚠ CUDA not available, will use CPU")
        
    return True


def main():
    print("="*60)
    print("ASM Function Naming - Quick Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Data", test_data),
        ("Projection", test_projection),
        ("Metrics", test_metrics),
        ("GPU", test_gpu)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ {name} failed with exception: {e}")
            results.append((name, False))
            
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
        
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Ready to train.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        

if __name__ == "__main__":
    main()
