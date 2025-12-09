import os 
import sys 
import json
from tqdm import tqdm
sys.path.insert(0, "/home/lab314/cjw/funame")
from config import PROJECT_ROOT, LOG_DIR
from utils.utils import setup_logger, get_json_files
from simhash import Simhash
import pandas as pd

logger = setup_logger("merge_asm_func")


if __name__ == "__main__":
    asm_proj_root = "resources/sym-dataset/unstripped_decompiled_data"
    output_csv = "resources/sym-dataset/asm_funcs.csv"
    merge_step = 1000
    logger.info(f"汇编函数JSON文件目录: {asm_proj_root}")
    logger.info(f"输出CSV文件路径: {output_csv}")
    logger.info(f"合并步长: {merge_step}")
    json_files = get_json_files(asm_proj_root)
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
            path_list = json_file.split('/')
            file_name = path_list[-1].split('.')[0]
            proj_name = path_list[-2]
            opti_level = path_list[-3]
            arch = path_list[-4]
            data = json.load(open(json_file, 'r', encoding='utf-8'))
            for k, v in data.items():
                function_name = k
                asm_func = v.get("assembly", "")
                asm_func = "\n".join(asm_func)
                key = proj_name + '$' + file_name  + '$' + function_name
                signature = proj_name  + '$' + function_name
                if len(asm_func.split('\n')) < 5 or function_name == 'main':
                    filter_func_num += 1
                    continue  # 汇编代码行数过短的函数和main函数
                all_data.append({
                    "key": key,
                    "signature": signature,
                    "file_name": file_name,
                    "function_name": function_name,
                    "asm_func": asm_func,
                    "opti_level": opti_level,
                    "arch": arch,
                })
        except Exception as e:
            logger.error(f"处理文件失败: {json_file} | 错误: {e}")
            continue
        if (idx + 1) % merge_step == 0 or idx == len(json_files) - 1:
            new_df = pd.DataFrame(all_data)
            if not os.path.exists(output_csv):
                new_df.to_csv(output_csv, index=False, encoding='utf-8')
            else:
                new_df.to_csv(output_csv, mode='a', index=False, header=False, encoding='utf-8')
            logger.info(f"已处理文件数: {idx + 1}, 已合并函数数: {len(all_data)}")
            all_data = []  # 清空已保存的数据以节省内存

    logger.info(f"共过滤main函数和汇编代码行数小于5的函数: {filter_func_num}个")
    logger.info("JSON文件合并完成。")