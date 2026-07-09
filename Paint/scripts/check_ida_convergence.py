"""IDA 收敛性检查工具。

用途：统计不同温度工况下 IDA 计算的成功、未收敛、倒塌和超时情况，并绘制汇总图。
做法：读取 ``Output_data`` 中各工况的 IDA 目录及警告日志，汇总为表格后绘图。
使用：先修改“用户编辑区”，再运行 ``python Paint/scripts/check_ida_convergence.py``。
"""

import os
import re

import pandas as pd
import matplotlib.pyplot as plt

from plot_common import (
    CELSIUS,
    celsius_label,
    configure_matplotlib as apply_common_style,
    fragility_file_name,
    normalize_temperature_label,
)

apply_common_style()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# =============================================================================
# 用户编辑区：模型、结果根目录和待统计温度
# =============================================================================
MODEL_NAME = "MC8_PFSDF"
BASE_DIR = os.path.join(os.getcwd(), "Output_data")
TEMPERATURES = ["-20", "-10", "0", "10", "20", "30", "40"]
# =============================================================================
# 用户编辑区结束
# =============================================================================


def analyze_ida_fast(base_path, temps, model_name="MC8_PFSDF"):
    summary_data = []
    failed_steps_data = []

    for temp in temps:
        ida_dir = os.path.join(base_path, f"{model_name}_{temp}", "MC8_IDA_data")
        log_file = os.path.join(ida_dir, "警告.log")

        total_runs = 0
        if os.path.exists(ida_dir):
            total_runs = sum(1 for item in os.listdir(ida_dir) if re.match(r"^\d+_\d+$", item))

        counts = {
            "Non-converged (no collapse)": 0,
            "Non-converged (collapse)": 0,
            "Timeout": 0,
        }

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
                    key = "Non-converged (no collapse)"
                elif "不收敛(倒塌)" in reason:
                    key = "Non-converged (collapse)"
                elif "超过最大计算时间" in reason:
                    key = "Timeout"
                else:
                    continue

                counts[key] += 1
                failed_steps_data.append(
                    {
                        f"Temperature ({CELSIUS})": temp,
                        "Ground motion": wave_num,
                        "Failed step": step_num,
                        "Warning type": key,
                    }
                )

        total_warnings = sum(counts.values())
        summary_data.append(
            {
                f"Temperature ({CELSIUS})": temp,
                "Total analyses": total_runs,
                "Converged": max(total_runs - total_warnings, 0),
                **counts,
            }
        )

    return pd.DataFrame(summary_data), pd.DataFrame(failed_steps_data)


def plot_comprehensive_summary(summary_df, steps_df, model_name="MC8_PFSDF"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    temperature_column = f"Temperature ({CELSIUS})"
    plot_df = summary_df.set_index(temperature_column)[
        ["Converged", "Non-converged (no collapse)", "Non-converged (collapse)", "Timeout"]
    ]
    plot_df.plot(kind="bar", stacked=True, ax=ax1, width=0.6, color=["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"])
    ax1.set_title(f"{model_name} IDA convergence summary", fontsize=14)
    ax1.set_xlabel(temperature_column, fontsize=12)
    ax1.set_ylabel("Number of analyses", fontsize=12)
    ax1.tick_params(axis="x", rotation=0)
    ax1.legend(title="Analysis status", loc="upper left", bbox_to_anchor=(1, 1))
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    if not steps_df.empty:
        steps_df.boxplot(column="Failed step", by=temperature_column, ax=ax2, grid=False)
        ax2.set_title("Failed-step distribution", fontsize=14)
        ax2.set_xlabel(temperature_column, fontsize=12)
        ax2.set_ylabel("Scaling step", fontsize=12)
        fig.suptitle("")
    else:
        ax2.text(0.5, 0.5, "No failed-step data", ha="center", va="center")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    summary_df, steps_df = analyze_ida_fast(BASE_DIR, TEMPERATURES, MODEL_NAME)
    print(summary_df.to_string(index=False))
    if not steps_df.empty:
        print(steps_df.head(15).to_string(index=False))
    plot_comprehensive_summary(summary_df, steps_df, MODEL_NAME)
