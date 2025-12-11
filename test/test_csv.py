import pandas as pd

csv_path = r"resources/sym-dataset/asm_funcs_with_strings.csv"

df = pd.read_csv(csv_path)
print(df.columns)
print("len: ", len(df))
print(df.head(20))