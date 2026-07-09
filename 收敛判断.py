import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体，确保图表正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体
plt.rcParams['axes.unicode_minus'] = False

def analyze_ida_fast(base_path, temps, model_name="MC8_PFSDF"):
    summary_data = []
    failed_steps_data = []  # 用于记录每一次发生警告的具体步数

    for temp in temps:
        ida_dir = os.path.join(base_path, f"{model_name}_{temp}", "MC8_IDA_data")
        log_file = os.path.join(ida_dir, "警告.log")

        # 1. 极速统计总分析次数 (仅读取目录名称，不打开文件，速度极快)
        total_runs = 0
        if os.path.exists(ida_dir):
            for item in os.listdir(ida_dir):
                # 匹配文件夹名格式如 "1_1", "44_15"
                if re.match(r'^\d+_\d+$', item):
                    total_runs += 1

        # 2. 读取警告日志文件
        counts = {
            "不收敛(未倒塌)": 0,
            "不收敛(倒塌)": 0,
            "计算超时": 0
        }

        if os.path.exists(log_file):
            # 兼容 gbk 和 utf-8，解决 0xb5 报错
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(log_file, 'r', encoding='gbk') as f:
                    lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 正则解析核心：提取 "X第Y次计算..."
                # group(1) 是地震波编号，group(2) 是调幅步数，group(3) 是具体的警告原因
                match = re.search(r'^(\d+)第(\d+)次计算(.*)', line)
                
                if match:
                    wave_num = int(match.group(1))
                    step_num = int(match.group(2))
                    reason = match.group(3).strip()

                    # 分类计数
                    if "不收敛(未倒塌)" in reason:
                        counts["不收敛(未倒塌)"] += 1
                    elif "不收敛(倒塌)" in reason:
                        counts["不收敛(倒塌)"] += 1
                    elif "超过最大计算时间" in reason:
                        counts["计算超时"] += 1

                    # 记录具体的失败步数数据，留作画图用
                    failed_steps_data.append({
                        "温度 (℃)": temp,
                        "地震波编号": wave_num,
                        "失败发生步数": step_num,
                        "警告类型": reason
                    })

        # 3. 计算成功收敛的次数 (总次数 - 警告次数)
        total_warnings = counts["不收敛(未倒塌)"] + counts["不收敛(倒塌)"] + counts["计算超时"]
        success_runs = total_runs - total_warnings if total_runs > 0 else 0

        summary_data.append({
            "温度 (℃)": temp,
            "总调幅分析次数": total_runs,
            "成功收敛次数": success_runs,
            "不收敛(未倒塌)": counts["不收敛(未倒塌)"],
            "不收敛(倒塌)": counts["不收敛(倒塌)"],
            "计算超时": counts["计算超时"]
        })

    # 将数据转换为 DataFrame
    summary_df = pd.DataFrame(summary_data)
    steps_df = pd.DataFrame(failed_steps_data)
    
    return summary_df, steps_df

def plot_comprehensive_summary(summary_df, steps_df, model_name="MC8_PFSDF"):
    """绘制双子图：左边看成功率堆叠图，右边看不收敛发生的步数分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ==========================================
    # 子图 1：总分析次数与成功/警告成分堆叠图
    # ==========================================
    plot_df = summary_df.set_index("温度 (℃)")[["成功收敛次数", "不收敛(未倒塌)", "不收敛(倒塌)", "计算超时"]]
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']
    
    plot_df.plot(kind='bar', stacked=True, ax=ax1, width=0.6, color=colors)
    ax1.set_title(f"{model_name} IDA分析收敛状态统计", fontsize=14)
    ax1.set_xlabel("温度 (℃)", fontsize=12)
    ax1.set_ylabel("时程分析总次数 (次)", fontsize=12)
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(title="运行结果", loc='upper left', bbox_to_anchor=(1, 1))

    # 在柱子上添加具体数值 (过滤掉0的值，避免数字重叠)
    for c in ax1.containers:
        labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in c]
        ax1.bar_label(c, labels=labels, label_type='center', color='white', weight='bold', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # ==========================================
    # 子图 2：发生警告的“运算步数”分布 (箱线图+散点)
    # ==========================================
    if not steps_df.empty:
        # 使用 pandas 自带的 boxplot 画基础分布
        steps_df.boxplot(column='失败发生步数', by='温度 (℃)', ax=ax2, grid=False, 
                         boxprops=dict(linewidth=2, color='blue'),
                         medianprops=dict(linewidth=2, color='red'))
        
        # 加上散点看具体密集程度
        for i, temp in enumerate(summary_df['温度 (℃)'].unique()):
            y = steps_df[steps_df['温度 (℃)'] == temp]['失败发生步数']
            x = [i + 1] * len(y)  # X轴坐标对应箱线图的位置
            ax2.scatter(x, y, alpha=0.3, color='gray', s=15, zorder=3)

        ax2.set_title("发生不收敛/超时的运算步数分布图", fontsize=14)
        ax2.set_xlabel("温度 (℃)", fontsize=12)
        ax2.set_ylabel("发生警告时的调幅步数 (Step)", fontsize=12)
        fig.suptitle('')  # 去除 boxplot 自动生成的居中标题
    else:
        ax2.text(0.5, 0.5, "未找到任何失败步数数据", ha='center', va='center')
        
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 配置基础路径和温度列表
    model_name = "MC8_PFSDF"
    BASE_DIR = r"G:\Opensees-PFSDF-Temp-Slfecentring\Output_data"
    TEMPERATURES = ['-20', '-10', '0', '10', '20', '30', '40']

    print("正在极速分析数据，请稍候...")
    summary_df, steps_df = analyze_ida_fast(BASE_DIR, TEMPERATURES, model_name)

    # 打印表格结果
    print("\n" + "="*75)
    print("各温度下 IDA 总分析次数与结果汇总：")
    print("="*75)
    print(summary_df.to_string(index=False))
    
    if not steps_df.empty:
        print("\n" + "="*75)
        print("【部分数据预览】警告发生时的波号与步数：")
        print("="*75)
        print(steps_df.head(15).to_string(index=False))
        print("...... (仅显示前15条)")

    # 绘制双子图
    plot_comprehensive_summary(summary_df, steps_df, model_name)