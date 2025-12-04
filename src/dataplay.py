import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class FuncAbstractDataset(Dataset):
    def __init__(self, df: pd.DataFrame, asm_tokenizer: AutoTokenizer, abstract_tokenizer: AutoTokenizer, ):
        self.df = df
        self.asm_tokenizer = asm_tokenizer
        self.abstract_tokenizer = abstract_tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        src_func = item['src_func']
        abstract = item['abstract']
        asm_inputs = self.asm_tokenizer(src_func, return_tensors='pt', truncation=True, padding='max_length', max_length=1024)
        abstract_inputs = self.abstract_tokenizer(abstract, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
        return {
            "asm_inputs": asm_inputs,
            "abstract_inputs": abstract_inputs
        }