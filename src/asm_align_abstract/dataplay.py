import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

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
        asm_inputs = self.asm_tokenizer([{'function': src_func}], return_tensors='pt', padding='max_length', max_length=64)
        asm_inputs['input_ids'] = asm_inputs['input_ids'].squeeze(0)
        asm_inputs['attention_mask'] = asm_inputs['attention_mask'].squeeze(0)
        asm_inputs['token_type_ids'] = asm_inputs['token_type_ids'].squeeze(0)
        abstract_inputs = self.abstract_tokenizer([abstract], return_tensors='pt', truncation=True, padding='max_length', max_length=256)
        abstract_inputs['input_ids'] = abstract_inputs['input_ids'].squeeze(0)
        abstract_inputs['attention_mask'] = abstract_inputs['attention_mask'].squeeze(0)
        abstract_inputs['token_type_ids'] = abstract_inputs['token_type_ids'].squeeze(0)
        return {
            "asm_inputs": asm_inputs,
            "abstract_inputs": abstract_inputs
        }
        
if __name__ == "__main__":
    data_path = 'resources/dataset/func_pairs_with_abstract.csv'
    df = pd.read_csv(data_path)
    asm_tokenizer = AutoTokenizer.from_pretrained('hustcw/clap-asm', trust_remote_code=True)
    abstract_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    dataset = FuncAbstractDataset(df, asm_tokenizer, abstract_tokenizer)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for batch in data_loader:
        asm_inputs = batch['asm_inputs']
        abstract_inputs = batch['abstract_inputs']
        
        asm_inputids = asm_inputs['input_ids']
        asm_attentionmask = asm_inputs['attention_mask']
        abstract_inputids = abstract_inputs['input_ids']
        abstract_attentionmask = abstract_inputs['attention_mask']
        pass