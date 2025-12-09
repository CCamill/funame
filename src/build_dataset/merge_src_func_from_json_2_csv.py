"""
将resources/datasets/source_funcs_treesitter目录下的所有json文件中的数据提取出来合并到一个csv文件，并对数据进行清洗和整理。
每个csv文件对应一个目标文件的所有函数汇编代码
"""
from simhash import Simhash, SimhashIndex
import sys
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

sys.path.insert(0, "/home/lab314/cjw/funame")
from config import PROJECT_ROOT, LOG_DIR
from utils.utils import setup_logger
logger = setup_logger("merg_src_to_csv")


def get_key_words(source_path, proj_name):
    source_path_list = source_path.split('/')
    proj_namea_idx = source_path_list.index(proj_name)
    key_words = source_path_list[proj_namea_idx + 1: ]

    return list(set(key_words))

def get_jseon_files(source_funcs_root):
    json_files = []
    for root, dirs, files in os.walk(source_funcs_root):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

if __name__ == "__main__":
    merg_step = 20000
    source_funcs_root = "resources/sym-dataset/source_funcs_treesitter"
    output_csv = "resources/sym-dataset/src_funcs.csv"
    logger.info(f"源函数JSON文件目录: {source_funcs_root}")
    logger.info(f"输出CSV文件路径: {output_csv}")
    
    json_files = get_jseon_files(source_funcs_root)
    logger.info(f"找到 {len(json_files)} 个JSON文件。")
    logger.info("开始合并JSON文件到CSV...")
    all_data = []
    process_bar = tqdm(enumerate(json_files), 
                       total=len(json_files),
                       desc="合并JSON文件", 
                       ncols=100)
    filter_func_num = 0
    for idx, json_file in process_bar:
        try:
            data = json.load(open(json_file, 'r', encoding='utf-8'))
            proj_name = os.path.basename(os.path.dirname(json_file))
            # 提取目标文件名
            file_name = os.path.basename(json_file).split('.')[0]
            for func in data.get("functions", []):
                function_name = func.get("function_name", "")
                src_func = func.get("body", "")
                function_define = func.get("signature", "")
                if function_name == 'main' or len(src_func.split('\n')) < 5:
                    filter_func_num += 1
                    continue  # main函数和源代码行数过短的函数
                key = proj_name + '$' + file_name  + '$' + function_name
                signature = proj_name  + '$' + function_name

                key_words = get_key_words(data["metadata"].get("source_file"), proj_name)
                all_data.append({
                    "key": key,
                    "signature": signature,
                    "file_name": file_name,
                    "function_name": function_name,
                    "src_func": src_func,
                })
            if (idx + 1) % merg_step == 0 or idx == len(json_files) - 1:
                new_data = pd.DataFrame(all_data)
                if not os.path.exists(output_csv):
                    new_data.to_csv(output_csv, index=False, encoding='utf-8')
                else:
                    new_data.to_csv(output_csv, mode='a', index=False, header=False, encoding='utf-8')
                logger.info(f"已处理文件数: {idx + 1}, 已合并函数数: {len(all_data)}")
                all_data = []  # 清空已保存的数据以节省内存
        except Exception as e:
            logger.info(f"处理文件 {json_file} 时出错: {str(e)}")
    logger.info(f"共过滤main函数和词数小于5的函数: {filter_func_num}个")
    logger.info("JSON文件合并完成。")