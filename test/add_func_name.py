import pandas as pd

if __name__ == "__main__":
    abstract_df = pd.read_csv(r"resources\dataset\src_funcs_absrtract_large.csv")
    src_df = pd.read_csv(r"resources\dataset\src_funcs_deduplicated_large.csv")
    len_df = len(abstract_df)
    all_data = []
    for i in range(len_df):
        ab = abstract_df.iloc[i]
        ab_src = ab['src_func']
        src = src_df.iloc[i]
        src_src = src['source_code']
        all_data.append({
            "key": src['key'],
            "src_func": ab_src,
            "abstract": ab['abstract']}
        )
    new_df = pd.DataFrame(all_data)
    new_df.to_csv(r"resources\dataset\src_funcs_with_abstract_large.csv", index=False, encoding='utf-8')