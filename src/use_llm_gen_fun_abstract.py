"""
使用LLM生成函数摘要
"""
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import pandas as pd
import time
from tqdm import tqdm
import logging
import argparse
import os


def matching_res_parser(response: str) -> dict:
    """
    解析LLM的响应, 提取分析结果和置信度。
    
    Args:
        response: LLM的完整响应文本
        
    Returns:
        dict: 包含'analysis'和'if_same_source'的字典
    """
    if_same_source = response.splitlines()[0].strip()
    
    return {
        "if_same_source": if_same_source
    }

def load_prompt_from_file(file_path: str) -> PromptTemplate:
    with open(file_path, 'r', encoding='utf-8') as file:
        template = file.read()
    return PromptTemplate(input_variables=["code", "example_code"], template=template)


def match_assembly_to_source(model_name, code: str, prompt_path: str) -> str:
    example_code = """
    for (int i = 0; i < n; i  ) {
        if (arr[i] > 0) {
            sum  = arr[i];
        }
    }
    """
    prompt = load_prompt_from_file(prompt_path)
    model = OllamaLLM(model=model_name, base_url="http://localhost:11434")

    chain = prompt | model
    result = chain.invoke({
        "code": code,
        "example_func": example_code
    })

    return result

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="LLM gen abstract")
    argparser.add_argument('--prompt_path', type=str, default=r"tmp/fun_abstract_tmp.md", help='Path to the prompt template file')
    argparser.add_argument('--data_path', type=str, default=r"resources/dataset/dataset_no_opti.csv", help='Path to the dataset CSV file')
    argparser.add_argument('--output_path', type=str, default=r"resources/dataset/func_pairs_with_abstract.csv")
    argparser.add_argument('--log_path', type=str, default=f"resources/logs/llm_gen_abstract{int(time.time())}.log", help='Path to the log file')
    args = argparser.parse_args()
    print("data_path: ", args.data_path)
    print("output_path: ", args.output_path)

    model_name = r"qwen2.5-coder:14b"
    df = pd.read_csv(args.data_path)
    print("start")
    pbar = tqdm(total=len(df),
                    ncols=150)
    all_data = []
    step = 100
    len_df = len(df)
    for i in range(1, len_df+1):
        item = df.iloc[i]
        src_func = item['src_func']
        result = match_assembly_to_source(model_name, src_func, args.prompt_path)
        all_data.append({
            "key": item['key'],
            "func_name": item['func_name'],
            "src_func": src_func,
            "asm_func": item['asm_func'],
            "abstract": result
        })
        if i % 100 == 0 or i == len_df-1:
            new_df = pd.DataFrame(all_data)
            if not os.path.exists(args.output_path):
                new_df.to_csv(args.output_path, index=False, encoding='utf-8')
            else:
                new_df.to_csv(args.output_path, mode='a', index=False, header=False, encoding='utf-8')
            all_data = []
            print(f"以处理数据项：{i}")
        pbar.update(1)
    pbar.close()
    print("done")
