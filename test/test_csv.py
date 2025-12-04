import pandas as pd

csv_path = r"resources/dataset/absrtract_large_src_funcs.csv"

df = pd.read_csv(csv_path)
print(df.columns)
print("len: ", len(df))
print(df.head(20))