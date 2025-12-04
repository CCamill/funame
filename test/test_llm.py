"""
判断大语言模型对汇编代码的理解能力
"""
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import pandas as pd
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import argparse
import os


def match_assembly_to_source(model_name, asm_func: str) -> str:
    prompt = "Analyze the following assembly code and understand its semantic meaning:\n"\
            "```asm\n{code}\n```\n"\
            "Semantic representation:"
    model = OllamaLLM(model=model_name, base_url="http://localhost:11434")

    chain = prompt | model
    result = chain.invoke({
        "asm_func": asm_func,
    })

    return result

if __name__ == "__main__":
    model_name = r"qwen2.5-coder:14b"
    asm_func = """
    0xe endbr64
    0x12 push    r13
    0x14 mov     r13, rdi
    0x17 push    r12
    0x19 push    rbp
    0x1a mov     rbp, rsi
    0x1d push    rbx
    0x1e mov     rdi, rbp
    0x21 sub     rsp, 88h
    0x28 mov     rsi, [rsi+8]
    0x2c mov     rax, fs:28h
    0x35 mov     [rsp+0A8h+var_30], rax
    0x3a xor     eax, eax
    0x3c call    skip_white; PIC mode
    0x41 mov     rsi, [rbp+8]
    0x45 mov     rdi, rbp
    0x48 call    parse_pdf_boolean; PIC mode
    0x4d test    rax, rax
    0x50 jnz     short loc_62
    0x52 lea     rdi, _LC0; ""A boolean value expected but not found.""...
    0x59 xor     eax, eax
    0x5b call    dpx_warning; PIC mode
    0x60 jmp     short loc_C3
    0x62 mov     rdi, rax
    0x65 mov     r12, rax
    0x68 call    pdf_boolean_value; PIC mode
    0x6d mov     rdi, r12
    0x70 xor     r12d, r12d
    0x73 movsx   ebx, al
    0x76 call    pdf_release_obj; PIC mode
    0x7b mov     esi, ebx
    0x7d mov     rdi, r13
    0x80 call    spc_set_linkmode; PIC mode
    0x85 mov     rsi, [rbp+8]
    0x89 mov     rdi, rbp
    0x8c call    skip_white; PIC mode
    0x91 dec     ebx
    0x93 jnz     short loc_F0
    0x95 mov     rax, [rbp+8]
    0x99 cmp     [rbp+0], rax
    0x9d jnb     short loc_F0
    0x9f lea     r12, [rsp+0A8h+var_A0]
    0xa4 mov     rdi, r12
    0xa7 call    transform_info_clear; PIC mode
    0xac mov     rsi, r12
    0xaf xor     ecx, ecx
    0xb1 mov     rdx, rbp
    0xb4 mov     rdi, r13
    0xb7 call    spc_util_read_dimtrns; PIC mode
    0xbc mov     r12d, eax
    0xbf test    eax, eax
    0xc1 jz      short loc_C9
    0xc3 or      r12d, 0FFFFFFFFh
    0xc7 jmp     short loc_F0
    0xc9 test    [rsp+0A8h+var_38], 4
    0xce jz      short loc_E4
    0xd0 movsd   xmm1, [rsp+0A8h+var_90]
    0xd6 movsd   xmm0, [rsp+0A8h+var_98]
    0xdc mov     rdi, r13
    0xdf call    spc_set_phantom; PIC mode
    0xe4 mov     rsi, [rbp+8]
    0xe8 mov     rdi, rbp
    0xeb call    skip_white; PIC mode
    0xf0 mov     rax, [rsp+0A8h+var_30]
    0xf5 sub     rax, fs:28h
    0xfe jz      short loc_105
    0x100 call    __stack_chk_fail; PIC mode
    0x105 add     rsp, 88h
    0x10c mov     eax, r12d
    0x10f pop     rbx
    0x110 pop     rbp
    0x111 pop     r12
    0x113 pop     r13
    0x115 retn
    """
    result = match_assembly_to_source(model_name, asm_func)
                    