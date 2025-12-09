"""
从常用第三方库构建构建数据集
包括个各别代码优化
批量多进程处理 .o 文件，保持目录结构
obj/项目/xxx.o -> ida_funcs/项目/xxx.csv
每个 .csv 文件包含函数名、汇编代码等信息
"""
import os
import sys
import csv
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
from tqdm import tqdm

sys.path.insert(0, "/home/lab314/cjw/funame")
from utils.utils import setup_logger, get_ida_path

logger = setup_logger("extract_asm_using_ida")

logger.info(f"IDA Pro 可执行文件路径: {get_ida_path()}")

def process_single_file_wrapper(args):
    """包装函数用于多进程处理"""
    arch, proj_name, opti, input_file = args
    return process_single_file(arch, proj_name, opti, input_file)

def process_single_file(arch, proj_name, opti, input_file):
    """处理单个文件，保持目录结构"""
    try:
        file_name = input_file.split('/')[-1] + '.csv'
        output_csv_path = os.path.join(r"resources/sym-dataset/ida_funcs/", arch, opti, proj_name, file_name)
        
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # 创建处理脚本 - 关键修复：等待分析和使用正确的API
        script_content = f'''import idc
import idautils
import idaapi
import ida_auto
import csv
import os
import time

# 输出文件路径
output_csv = r"{output_csv_path}"
input_filename = r"{os.path.basename(input_file)}"

print("开始处理:", input_filename)

# 等待IDA完成自动分析
print("等待自动分析完成...")
ida_auto.auto_wait()

# 给分析更多时间
time.sleep(2)

functions_data = []

# 获取所有段中的函数
for seg_ea in idautils.Segments():
    seg_start = idc.get_segm_start(seg_ea)
    seg_end = idc.get_segm_end(seg_ea)
    
    # 遍历段中的所有函数
    for func_ea in idautils.Functions(seg_start, seg_end):
        try:
            func_name = idc.get_func_name(func_ea)
            func_start = func_ea
            func_end = idc.find_func_end(func_ea)
            
            if func_end == idc.BADADDR:
                continue
                
            instructions = []
            addr = func_start
            
            # 收集函数的所有指令
            while addr < func_end and addr != idc.BADADDR:
                asm = idc.GetDisasm(addr)
                if asm:
                    instr_line = asm
                    instructions.append(instr_line)
                addr = idc.next_head(addr, func_end)
            
            # 准备函数数据
            func_data = {{
                'function_name': func_name,
                'full_define': " | ".join(instructions),
                'start_addr': hex(func_start),
                'source_file': input_filename
            }}
            functions_data.append(func_data)
            print(f"找到函数: {{func_name}} ({{len(instructions)}} 条指令)")
            
        except Exception as e:
            print("处理函数时出错:", str(e))
            continue

print(f"总共找到 {{len(functions_data)}} 个函数")

# 写入CSV文件
try:
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        if functions_data:
            fieldnames = ['function_name', 'full_define', 'start_addr', 'source_file']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(functions_data)
            print("成功写入", len(functions_data), "个函数到", output_csv)
        else:
            print("未找到任何函数")
            # 创建包含错误信息的文件
            with open(output_csv, 'w') as f:
                f.write("function_name,full_define,start_addr,source_file\\n")
                f.write("# No functions found in this file\\n")
except Exception as e:
    print("写入文件时出错:", str(e))
    # 即使出错也创建文件记录错误
    with open(output_csv, 'w') as f:
        f.write("function_name,full_define,start_addr,source_file\\n")
        f.write(f"# Error: {{str(e)}}\\n")

print("处理完成")

# 正确退出IDA
idaapi.qexit(0)
'''
        
        # 将脚本写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_file = f.name
        
        # 执行 IDA 命令 - 增加超时时间
        cmd = [
            get_ida_path(), 
            "-A",
            f"-S{script_file}",
            input_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # 清理临时脚本
        if os.path.exists(script_file):
            os.remove(script_file)
        
        if result.returncode == 0 and os.path.exists(output_csv_path):
            return (input_file, True, f"成功: {output_csv_path}")
        else:
            # 输出错误信息
            error_msg = result.stderr or "未知错误"
            return (input_file, False, f"IDA处理失败: {error_msg}")
            
    except subprocess.TimeoutExpired:
        return (input_file, False, "处理超时(300秒)")
    except Exception as e:
        return (input_file, False, f"异常: {str(e)}")

def find_all_exe_files(exe_directory):
    exe_files = []
    for arch in os.listdir(exe_directory):
        for opti in os.listdir(os.path.join(exe_directory, arch)):
            for proj in os.listdir(os.path.join(exe_directory, arch, opti)):
                for file in os.listdir(os.path.join(exe_directory, arch, opti, proj)):
                    arg = (arch, proj, opti, file)
                    exe_files.append(arg)
    return exe_files

def mutiprocess_batch_process_with_structure(exe_directory, max_workers=8):
    """
    批量处理并保持目录结构
    TODO: 有BUG，需完善
    """
    exe_files = find_all_exe_files(exe_directory)
    # 准备任务参数
    tasks = exe_files

    # 多进程处理
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers or multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(process_single_file_wrapper, task): task for task in tasks}
        for future in as_completed(futures):
            input_file, success, message = future.result()
            status = "✓" if success else "✗"
            filename = os.path.basename(input_file)
            logger.info(f"{status} {filename}: {message}")
            if success:
                success_count += 1
    logger.info(f"成功处理 {success_count} 个文件")
    logger.info(f"失败处理 {len(exe_files) - success_count} 个文件")

def single_process_batch_process_with_structure(exe_directory):
    exe_files = find_all_exe_files(exe_directory)
    success_count = 0
    fail_count = 0
    process_bar = tqdm(enumerate(exe_files),
                       total=len(exe_files),
                       desc="处理文件",
                       ncols=100)
    for i, exe_file in process_bar:
        path_list = exe_file.split('/')
        proj_name = path_list[-2]
        opti = path_list[-3]
        arch = path_list[-4]
        input_file, success, message = process_single_file(arch, proj_name, opti, exe_file)
        success_count += int(success)
        fail_count += int(not success)
        process_bar.set_postfix(success=success_count, fail=fail_count)
    process_bar.close()
    logger.info(f"成功处理 {success_count} 个文件")
    logger.info(f"失败处理 {fail_count} 个文件")

if __name__ == "__main__":
    exe_directory = r"resources/sym-dataset/binaries"
    single_process_batch_process_with_structure(exe_directory)
            
