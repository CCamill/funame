"""
对func_pairs_with_strings.csv文件进行数据清洗,删除asm_func_len小于50和大于200的行,
如果数据项没有asm_func_len列,则删除该行
"""
import pandas as pd

if __name__ == "__main__":
    csv_path = "resources/sym-dataset/func_pairs_with_strings.csv"
    output_csv_path = "resources/sym-dataset/func_pairs_with_strings_cleaned.csv"
    df = pd.read_csv(csv_path)
    print("start cleaning...")
    print("len: ", len(df))
    # 如果不存在asm_func_len列，则根据asm_func列计算
    if 'asm_func_len' not in df.columns:
        if 'asm_func' in df.columns:
            df['asm_func_len'] = df['asm_func'].apply(lambda x: len(str(x).split('|')) if pd.notna(x) else 0)
        else:
            print("错误: CSV文件中既没有asm_func_len列，也没有asm_func列")
            exit(1)
    
    # 删除asm_func_len为NaN的行
    df = df[df['asm_func_len'].notna()]
    # 过滤长度在50到200之间的行
    df = df[df['asm_func_len'] >= 50]
    df = df[df['asm_func_len'] <= 200]
    print("after cleaning, len: ", len(df))
    df.to_csv(output_csv_path, index=False, encoding='utf-8')