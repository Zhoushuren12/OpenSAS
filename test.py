import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体，确保图表正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体
plt.rcParams['axes.unicode_minus'] = False

def analyze_ida_status(base_path, temps, model_name="MC8_PFSDF"):
    summary_data = []

    for temp in temps:
        # 构造对应温度的 IDA_data 目录
        ida_dir = os.path.join(base_path, f"{model_name}_{temp}", "MC8_IDA_data")
        
        counts = {
            "收敛完成 (Status 1)": 0,
            "不收敛 (Status 2)": 0,
            "计算超时 (Status 3)": 0,
            "状态文件缺失": 0
        }
        total_runs = 0

        if os.path.exists(ida_dir):
            # 遍历 IDA_data 下的所有文件和文件夹
            for item in os.listdir(ida_dir):
                # 使用正则匹配类似 "1_1", "44_15" 这样的波和调幅步文件夹
                if re.match(r'^\d+_\d+$', item):
                    folder_path = os.path.join(ida_dir, item)
                    
                    if os.path.isdir(folder_path):
                        total_runs += 1
                        status_file = os.path.join(folder_path, "Status.dat")
                        
                        if os.path.exists(status_file):
                            try:
                                # 读取 Status.dat 里的数字
                                with open(status_file, 'r', encoding='utf-8') as f:
                                    status_val = f.read().strip()
                                    
                                    if status_val == '1':
                                        counts["收敛完成 (Status 1)"] += 1
                                    elif status_val == '2':
                                        counts["不收敛 (Status 2)"] += 1
                                    elif status_val == '3':
                                        counts["计算超时 (Status 3)"] += 1
                                    else:
                                        counts["状态文件缺失"] += 1
                            except Exception as e:
                                counts["状态文件缺失"] += 1
                        else:
                            counts["状态文件缺失"] += 1
        else:
            print(f"⚠️ 提示: 未找到文件夹 {ida_dir}")

        # 汇总该温度的数据
        summary_data.append({
            "温度 (℃)": temp,
            "收敛完成 (Status 1)": counts["收敛完成 (Status 1)"],
            "不收敛 (Status 2)": counts["不收敛 (Status 2)"],
            "计算超时 (Status 3)": counts["计算超时 (Status 3)"],
            "状态文件缺失": counts["状态文件缺失"],
            "总调幅分析次数": total_runs
        })

    # 转换为 DataFrame
    df = pd.DataFrame(summary_data)
    return df

def plot_status_summary(df, model_name="MC8_PFSDF"):
    """绘制堆叠柱状图，直观展示各状态占比"""
    # 提取需要绘图的列，并将温度设为索引
    plot_df = df.set_index("温度 (℃)")[["收敛完成 (Status 1)", "不收敛 (Status 2)", "计算超时 (Status 3)"]]
    
    # 使用堆叠柱状图 (绿色代表成功，红色代表不收敛，橙色代表超时)
    colors = ['#2ca02c', '#d62728', '#ff7f0e']
    ax = plot_df.plot(kind='bar', stacked=True, figsize=(10, 6), width=0.6, color=colors)
    
    plt.title(f"{model_name} 各温度下 IDA 时程分析总体收敛状态统计", fontsize=15)
    plt.xlabel("温度 (℃)", fontsize=12)
    plt.ylabel("时程分析总次数 (次)", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Status 状态", loc='upper left', bbox_to_anchor=(1, 1))
    
    # 在柱子上添加具体的数值标签 (只标注大于0的数值)
    for c in ax.containers:
        # 使用自定义的格式化函数，过滤掉0值
        labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 配置基础路径和温度列表
    model_name = "MC8_PFSDF"
    BASE_DIR = r"G:\Opensees-PFSDF-Temp\Output_data"
    TEMPERATURES = ['-20', '-10', '0', '10', '20', '30', '40']

    # 1. 执行分析，读取所有 Status.dat
    print("正在遍历文件夹并读取 Status.dat，请稍候...")
    summary_df = analyze_ida_status(BASE_DIR, TEMPERATURES, model_name)

    # 2. 打印表格结果
    print("\n" + "="*70)
    print("各温度下时程分析 (调幅步) 收敛状态汇总：")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("\n")

    # 3. 绘制统计图
    plot_status_summary(summary_df, model_name)