import logging
import os
import shutil
import sys
from datetime import datetime
from typing import List
sys.path.insert(0, "/home/lab314/cjw/funame")
from config import LOG_DIR

def setup_logger(name: str) -> logging.Logger:
    log_path = os.path.join(LOG_DIR, f"{name}--{datetime.now().strftime("%Y%m%d_%H%M%S")}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def should_process_file(filename: str, supported_extensions: set) -> bool:
    """检查文件是否应该被处理。
    
    过滤规则：
    - 仅处理扩展名在 supported_extensions 内的文件
    - 过滤以 "test_" 开头的测试文件
    
    Args:
        filename: 文件名
        supported_extensions: 支持的文件扩展名集合
        
    Returns:
        如果文件应该被处理返回True，否则返回False
    """
    basename = os.path.basename(filename)
    if basename.startswith("test_"):
        return False
    _, ext = os.path.splitext(basename)
    return ext in supported_extensions

def map_input_to_output(input_base: str, output_base: str, file_path: str) -> str:
    """将输入源码路径映射到输出JSON路径。
    
    Args:
        input_base: 输入基础目录
        output_base: 输出基础目录
        file_path: 输入文件路径
        
    Returns:
        输出JSON文件路径
        
    Raises:
        ValueError: 当无法从路径解析库名时
    """
    rel_path = os.path.relpath(file_path, input_base)
    parts = rel_path.split(os.sep)
    if not parts:
        raise ValueError(f"无法从路径解析库名: {file_path}")
    lib_name = parts[0]
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join(output_base, lib_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{base_name}.json")
    return out_file


def get_json_files(root_dir: str) -> List[str]:
    """获取指定目录下的所有JSON文件"""
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def get_ida_path():
    """获取 IDA Pro 可执行文件路径"""
    # Linux 常见路径
    possible_paths = [
        "/opt/ida/ida64",
        "/usr/local/bin/ida64", 
        "/home/username/ida/ida64",
        "ida64",
        "/home/lab314/sjj/tools/idapro-9.0/idat64"  # 修改为你的实际路径
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 如果找不到，尝试which命令
    ida_path = shutil.which("ida64") or shutil.which("ida")
    if ida_path:
        return ida_path
    
    raise Exception("未找到IDA Pro可执行文件，请检查安装路径")