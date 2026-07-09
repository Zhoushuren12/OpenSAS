import os
import re

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def analyze_ida_fast(base_path, temps, model_name="MC8_PFSDF"):
    summary_data = []
    failed_steps_data = []

    for temp in temps:
        ida_dir = os.path.join(base_path, f"{model_name}_{temp}", "MC8_IDA_data")
        log_file = os.path.join(ida_dir, "警告.log")

        total_runs = 0
        if os.path.exists(ida_dir):
            total_runs = sum(1 for item in os.listdir(ida_dir) if re.match(r"^\d+_\d+$", item))

        counts = {"不收敛(未倒塌)": 0, "不收敛(倒塌)": 0, "计算超时": 0}

        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(log_file, "r", encoding="gbk") as f:
                    lines = f.readlines()

            for line in lines:
                line = line.strip()
                match = re.search(r"^(\d+)第(\d+)次计算(.*)", line)
                if not match:
                    continue

                wave_num = int(match.group(1))
                step_num = int(match.group(2))
                reason = match.group(3).strip()

                if "不收敛(未倒塌)" in reason:
                    key = "不收敛(未倒塌)"
                elif "不收敛(倒塌)" in reason:
                    key = "不收敛(倒塌)"
                elif "超过最大计算时间" in reason:
                    key = "计算超时"
                else:
                    continue

                counts[key] += 1
                failed_steps_data.append(
                    {"温度 (℃)": temp, "地震波编号": wave_num, "失败发生步数": step_num, "警告类型": reason}
                )

        total_warnings = sum(counts.values())
        summary_data.append(
            {
                "温度 (℃)": temp,
                "总调幅分析次数": total_runs,
                "成功收敛次数": max(total_runs - total_warnings, 0),
                **counts,
            }
        )

    return pd.DataFrame(summary_data), pd.DataFrame(failed_steps_data)


def plot_comprehensive_summary(summary_df, steps_df, model_name="MC8_PFSDF"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    plot_df = summary_df.set_index("温度 (℃)")[
        ["成功收敛次数", "不收敛(未倒塌)", "不收敛(倒塌)", "计算超时"]
    ]
    plot_df.plot(kind="bar", stacked=True, ax=ax1, width=0.6, color=["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"])
    ax1.set_title(f"{model_name} IDA分析收敛状态统计", fontsize=14)
    ax1.set_xlabel("温度 (℃)", fontsize=12)
    ax1.set_ylabel("时程分析总次数", fontsize=12)
    ax1.tick_params(axis="x", rotation=0)
    ax1.legend(title="运行结果", loc="upper left", bbox_to_anchor=(1, 1))
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    if not steps_df.empty:
        steps_df.boxplot(column="失败发生步数", by="温度 (℃)", ax=ax2, grid=False)
        ax2.set_title("不收敛/超时发生步数分布", fontsize=14)
        ax2.set_xlabel("温度 (℃)", fontsize=12)
        ax2.set_ylabel("调幅步数", fontsize=12)
        fig.suptitle("")
    else:
        ax2.text(0.5, 0.5, "未找到失败步数数据", ha="center", va="center")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_name = "MC8_PFSDF"
    base_dir = os.path.join(os.getcwd(), "Output_data")
    temperatures = ["-20", "-10", "0", "10", "20", "30", "40"]

    summary_df, steps_df = analyze_ida_fast(base_dir, temperatures, model_name)
    print(summary_df.to_string(index=False))
    if not steps_df.empty:
        print(steps_df.head(15).to_string(index=False))
    plot_comprehensive_summary(summary_df, steps_df, model_name)
