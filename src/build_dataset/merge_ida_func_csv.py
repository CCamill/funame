"""
将resources/sym-dataset/ida_funcs目录下的所有csv文件合并为一个csv文件，并对数据进行清洗和整理。
每个csv文件对应一个目标文件的所有函数汇编代码
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import re
import cxxfilt
from tqdm import tqdm
import traceback
sys.path.insert(0, "/home/lab314/cjw/funame")

from utils.utils import setup_logger
logger = setup_logger("merge_ida_func_csv")

def is_mangled(name):
    """判断函数名是否经过名称修饰"""
    # C++ 名称修饰的常见特征
    patterns = [
        r'^_Z',                    # Itanium C++ ABI
        r'^\?',                    # Microsoft修饰
        r'^__Z',                   # 某些系统的Itanium变体
    ]
    
    for pattern in patterns:
        if re.match(pattern, name):
            return True
    return False

def demangle_function(name):
    """还原名称修饰的函数名"""
    try:
        return cxxfilt.demangle(name)
    except:
        return name  # 如果还原失败，返回原名称

def extract_function_name_advanced(demangled_name):
    """高级函数名提取，处理各种复杂情况"""
    
    # 移除尾部的参数列表
    if '(' in demangled_name and demangled_name.endswith(')'):
        # 找到最后一个括号对
        stack = []
        for i, char in enumerate(demangled_name):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    start = stack.pop()
                    if not stack:  # 最外层的括号对
                        base_name = demangled_name[:start].strip()
                        break
        else:
            base_name = demangled_name.split('(')[0].strip()
    else:
        base_name = demangled_name
    
    # 提取函数名部分（去掉返回类型）
    patterns = [
        # 匹配: 返回类型 函数名
        r'^[a-zA-Z_][a-zA-Z0-9_:]+\s+([a-zA-Z_][a-zA-Z0-9_]+)$',
        # 匹配: 函数名 (无返回类型的情况，如构造函数)
        r'^([a-zA-Z_][a-zA-Z0-9_]+)$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, base_name.strip())
        if match:
            return match.group(1)
    
    # 如果以上都不匹配，尝试分割空格取最后一部分
    parts = base_name.strip().split()
    if parts:
        return parts[-1]  # 通常函数名在最后
    
    return base_name



def get_csv_files(source_funcs_root):
    csv_files = []
    for root, dirs, files in os.walk(source_funcs_root):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def merage_ida_csv():
    ida_funcs_root = "resources/sym-dataset/ida_funcs"
    csv_files = get_csv_files(ida_funcs_root)
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, comment='#')
            if df.empty:
                continue
            proj_name = os.path.basename(os.path.dirname(csv_file))
            # 提取目标文件名
            file_name = os.path.basename(csv_file).split('.')[0]
            
            for _, row in df.iterrows():
                function_name = row.get("function_name", "")
                assembly_code = row.get("full_define", "")
                if assembly_code==np.nan:
                    continue  # 跳过汇编代码过短的函数
                if len(assembly_code) < 10:
                    continue  # 跳过汇编代码过短的函数
                start_addr = row.get("start_addr", "")
                signature = proj_name + '$' + file_name  + '$' + function_name
                all_data.append({
                    "signature": signature,
                    "function_name": function_name,
                    "assembly_code": assembly_code,
                    "start_addr": start_addr,
                })
        except Exception as e:
            logger.error(f"处理文件 {csv_file} 时出错: {str(e)}")
    
    logger.info(f"总共合并了 {len(all_data)} 个函数")
    # 创建DataFrame并保存为合并后的CSV文件
    merged_df = pd.DataFrame(all_data)
    output_csv = "resources/sym-dataset/ida_funcs.csv"
    merged_df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"合并完成，输出文件: {output_csv}")

def merage_ida_csv_with_opti_arch():
    ida_funcs_root = "resources/sym-dataset/ida_funcs"
    output_csv = "resources/sym-dataset/asm_funcs_with_strings.csv"
    csv_files = get_csv_files(ida_funcs_root)
    merge_step = 1000
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    logger.info(f"输出CSV文件路径: {output_csv}")
    logger.info(f"合并步长: {merge_step}")
    all_data = []
    process_bar = tqdm(enumerate(csv_files), total=len(csv_files), desc="合并CSV文件", ncols=100)
    for idx, csv_file in process_bar:
        try:
            df = pd.read_csv(csv_file, comment='#')
            if df.empty:
                continue
            path_list = csv_file.split(r'/')
            file_name = path_list[-1].split('.')[0]
            proj_name = path_list[-2]
            opti_level = path_list[-3]
            arch = path_list[-4]
            for _, row in df.iterrows():
                function_name = row.get("function_name", "")
                # if is_mangled(function_name):
                #     function_name = demangle_function(function_name)
                #     function_name = extract_function_name_advanced(function_name)
                asm_func = row.get("full_define", "")
                if asm_func==np.nan:
                    continue  # 跳过汇编代码过短的函数
                asm_func_len = len(asm_func.split('|'))
                if asm_func_len < 5:
                    continue  # 跳过汇编代码过短的函数

                key = proj_name + '$' + file_name  + '$' + function_name
                signature = proj_name + '$' + function_name
                all_data.append({
                    "key": key,
                    "signature": signature,
                    "file_name": file_name,
                    "function_name": function_name,
                    "asm_func": asm_func,
                    "arch": arch,
                    "opti_level": opti_level,
                    "asm_func_len": asm_func_len,
                })
            if (idx + 1) % merge_step == 0 or idx == len(csv_files) - 1:
                new_data = pd.DataFrame(all_data)
                if not os.path.exists(output_csv):
                    new_data.to_csv(output_csv, index=False, encoding='utf-8')
                else:
                    new_data.to_csv(output_csv, mode='a', index=False, header=False, encoding='utf-8')
                logger.info(f"已处理文件数: {idx + 1}, 已合并函数数: {len(all_data)}")
                all_data = []  # 清空已保存的数据以节省内存
        except Exception as e:
            tb = sys.exc_info()[2]
            line_number = tb.tb_lineno
            logger.error(f"处理文件 {csv_file} 时出错: {str(e)} 第 {line_number} 行")
    logger.info(f"总共合并了 {len(all_data)} 个函数")
    logger.info(f"合并完成，输出文件: {output_csv}")

if __name__ == "__main__":
    merage_ida_csv_with_opti_arch()