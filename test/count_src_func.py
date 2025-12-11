from collections import defaultdict
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
logger = setup_logger("count_src_func")


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
    source_funcs_root = "resources/sym-dataset/source_funcs_treesitter"
    logger.info(f"源函数JSON文件目录: {source_funcs_root}")
    
    json_files = get_jseon_files(source_funcs_root)
    logger.info(f"找到 {len(json_files)} 个JSON文件。")
    process_bar = tqdm(enumerate(json_files), 
                       total=len(json_files),
                       desc="统计函数数量", 
                       ncols=100)
    proj_funcs = defaultdict(int)
    for idx, json_file in process_bar:
        data = json.load(open(json_file, 'r', encoding='utf-8'))
        proj_name = os.path.basename(os.path.dirname(json_file))
        for func in data.get("functions", []):
            function_name = func.get("function_name", "")
            src_func = func.get("body", "")
            if function_name == 'main' or len(src_func.split('\n')) < 5:
                continue  # main函数和源代码行数过短的函数
            proj_funcs[proj_name] += 1
    logger.info(f"统计结果:")
    for proj_name, func_num in proj_funcs.items():
        logger.info(f"{proj_name}: {func_num}个函数")
    logger.info(f"总计: {sum(proj_funcs.values())}个函数")