"""
This script is used to split the samples into train and eval sets.
"""
import os
import random
import shutil
import pandas as pd

if __name__ == "__main__":
    input_csv_path = "resources/sym-dataset/func_pairs_with_strings_samples.csv"
    output_train_csv_path = "resources/sym-dataset/func_pairs_with_strings_train.csv"
    output_eval_csv_path = "resources/sym-dataset/func_pairs_with_strings_eval.csv"
    split_ratio = 0.8
    df = pd.read_csv(input_csv_path)
    print("len: ", len(df))
    train_df = df.sample(frac=split_ratio, random_state=42)
    eval_df = df.drop(train_df.index)
    print("train len: ", len(train_df))
    print("eval len: ", len(eval_df))
    train_df.to_csv(output_train_csv_path, index=False)
    eval_df.to_csv(output_eval_csv_path, index=False)
    print("done")