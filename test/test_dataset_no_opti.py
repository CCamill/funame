import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(r"resources\dataset\dataset_no_opti.csv")
    print(df.columns)
    
    ab_df = pd.read_csv(r"resources\dataset\src_funcs_absrtract_large.csv")
    print(ab_df.columns)
    for i in range(len(df)):
        src_func = df.at[i, 'src_func']
        abstract = ab_df[ab_df['src_func'] == src_func]['abstract'].values
        df.at[i, 'abstract'] = ab_df.at[i, 'abstract']
    print(df.columns)