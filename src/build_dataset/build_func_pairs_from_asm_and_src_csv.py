import sys
sys.path.insert(0, "/home/lab314/cjw/funame")
from settings import PROJECT_ROOT, LOG_DIR
from utils.utils import setup_logger
logger = setup_logger("build_func_pairs_from_asm_and_src_csv")
import pandas as pd
import os
from tqdm import tqdm

if __name__ == "__main__":
    asm_csv_path = "resources/sym-dataset/asm_funcs_with_strings.csv"
    src_csv_path = "resources/sym-dataset/src_funcs.csv"
    output_csv_path = "resources/sym-dataset/func_pairs_with_strings.csv"
    merge_step = 1000
    logger.info(f"汇编函数CSV文件路径: {asm_csv_path}")
    logger.info(f"源函数CSV文件路径: {src_csv_path}")
    logger.info(f"输出CSV文件路径: {output_csv_path}")
    logger.info(f"合并步长: {merge_step}")
    asm_df = pd.read_csv(asm_csv_path)
    src_df = pd.read_csv(src_csv_path)
    logger.info(f"asm df len: {len(asm_df)}")
    logger.info(f"src df len: {len(src_df)}")
    all_data = []
    process_bar = tqdm(range(len(src_df)), desc="构建函数对", ncols=100)
    for idx in process_bar:
        src_func = src_df.iloc[idx]['src_func']
        src_signature = src_df.iloc[idx]['signature']
        asm_item = asm_df[asm_df['signature'] == src_signature]
        if len(asm_item) == 0:
            continue
        asm_funcs = asm_item['asm_func'].values
        opti_level = asm_item['opti_level'].values
        arch = asm_item['arch'].values
        for i in range(len(asm_funcs)):
            all_data.append({
                "key": src_df.iloc[idx]['key'],
                "signature": src_signature,
                "function_name": src_df.iloc[idx]['function_name'],
                "src_func": src_func,
                "asm_func": asm_funcs[i],
                "opti_level": opti_level[i],
                "arch": arch[i],
            })
        if (idx + 1) % merge_step == 0 or idx == len(src_df) - 1:
            new_df = pd.DataFrame(all_data)
            if not os.path.exists(output_csv_path):
                new_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            else:
                new_df.to_csv(output_csv_path, mode='a', index=False, header=False, encoding='utf-8')
            logger.info(f"已处理数据项: {idx + 1}, 已合并函数数: {len(all_data)}")
            all_data = []
    logger.info("CSV文件合并完成。")