"""
将func_pairs_with_strings.csv文件拆分为train.csv和eval.csv
func_pairs_with_strings.csv中包含每个函数的16个版本（4优化等级，4架构），每个版本包含一个汇编函数和两个源函数
每个函数随机选择4个版本作为样本
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def sample_functions_from_csv(input_file, output_file, n_samples=4, random_state=None, function_id_col='signature'):
    """
    从CSV文件中为每个函数随机选择指定数量的版本
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    n_samples: 每个函数要选择的版本数（默认4个）
    random_state: 随机种子，用于可重复性
    function_id_col: 标识函数名的列名
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 设置随机种子（如果提供）
    if random_state is not None:
        np.random.seed(random_state)
    
    # 为每个函数随机选择指定数量的版本
    sampled_dfs = []
    save_steps = 1000
    
    for idx, (func_name, group) in tqdm(
                                    enumerate(df.groupby(function_id_col)), 
                                    total=len(df.groupby(function_id_col)), 
                                    desc="Sampling functions", 
                                    ncols=100
                                ):
        # 检查是否有足够的版本
        if len(group) >= n_samples:
            # 随机选择n_samples个版本
            sampled = group.sample(n=n_samples, random_state=random_state)
        else:
            sampled = group.copy()
        
        sampled_dfs.append(sampled)
        if (idx + 1) % save_steps == 0 or idx == len(df) - 1:
            new_df = pd.concat(sampled_dfs, ignore_index=True)
            if not os.path.exists(output_file):
                new_df.to_csv(output_file, index=False, encoding='utf-8')
            else:
                new_df.to_csv(output_file, mode='a', index=False, header=False, encoding='utf-8')
            sampled_dfs = []
            print(f"已处理函数数: {idx + 1}, 已保存函数数: {len(new_df)}")

# 使用示例
if __name__ == "__main__":
    # 方法1: 基本使用
    sample_functions_from_csv(
        input_file='resources/sym-dataset/func_pairs_with_strings_cleaned.csv',
        output_file='resources/sym-dataset/func_pairs_with_strings_samples.csv',
        n_samples=4,
        random_state=42  # 设置随机种子确保可重复性
    )