import os

# 原始文件路径
original_file = r'models\MC8_PFSDF.py'

# T 值的列表
T_values = [-20, -10, 0, 10, 20, 30, 40]
# T_values = [-20, 0, 20, 40]

# 获取文件名的基本部分
file_base_name = os.path.splitext(original_file)[0]

# 循环每个 T 值，生成对应的文件
for T in T_values:
    # 生成新文件名
    new_file_name = f"{file_base_name}_{T}.py"

    # 读取原始文件内容
    with open(original_file, 'r',encoding='utf-8-sig') as file:
        lines = file.readlines()

    # 查找并更新定义温度的那一行
    for idx, line in enumerate(lines):
        if line.strip().startswith('T ='):
            lines[idx] = f"    T = {T}\n"
            break
    else:
        raise RuntimeError('未在原始文件中找到 "T =" 定义，无法完成替换')

    # 写入新文件
    with open(new_file_name, 'w') as file:
        file.writelines(lines)

    print(f"已生成文件 {new_file_name}")
