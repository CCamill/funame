import idaapi, idc, idautils, csv, os

functions_data = []
for func_ea in idautils.Functions():
    func_name = idc.get_func_name(func_ea)
    func_start = func_ea
    func_end = idc.find_func_end(func_ea)
    if func_end != idc.BADADDR:
        instructions = []
        addr = func_start
        while addr < func_end and addr != idc.BADADDR:
            asm = idc.GetDisasm(addr)
            if asm: 
                instructions.append(f"{hex(addr)} {asm}")
            addr = idc.next_head(addr, func_end)
        
        functions_data.append({
            'function_name': func_name,
            'full_define': " | ".join(instructions),
            'start_addr': hex(func_start),
            'source_file': "threadtest5.o"
        })

# 确保输出目录存在
os.makedirs(os.path.dirname("resources/sym-dataset/ida_funcs/sqlite/threadtest5.csv"), exist_ok=True)

with open("resources/sym-dataset/ida_funcs/sqlite/threadtest5.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=['function_name', 'full_define', 'start_addr', 'source_file'])
    writer.writeheader()
    writer.writerows(functions_data)

print("处理完成: resources/sym-dataset/ida_funcs/sqlite/threadtest5.csv")
idaapi.qexit(0)
