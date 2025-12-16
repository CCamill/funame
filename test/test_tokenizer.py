from transformers import AutoModel, AutoTokenizer
import pandas as pd
model_name_or_path = "hustcw/clap-asm"
encoder = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

train_path = "resources/sym-dataset/func_pairs_with_strings_train.csv"
df = pd.read_csv(train_path)
for index, row in df.iterrows():
    asm_func = row["asm_func"]
    asm_encoding = tokenizer([{"function": asm_func}], return_tensors="pt", padding="max_length", max_length=768)    
    asm_input = asm_encoding["input_ids"].squeeze(0)
    asm_attention_mask = asm_encoding["attention_mask"].squeeze(0)
    asm_len = asm_attention_mask.sum(dim=0)
    continue