import pandas as pd

csv_path = r"resources/sym-dataset/func_pairs_with_strings_samples.csv"

df = pd.read_csv(csv_path)
print(df.columns)
print("len: ", len(df))
print(df.head(20))