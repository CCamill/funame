import pandas as pd

csv_path = r"resources\dataset\dataset_no_opti.csv"

df = pd.read_csv(csv_path)
print(df.columns)
print("len: ", len(df))
funcs = df['src_func'].to_list()
print("func len: ", len(funcs))