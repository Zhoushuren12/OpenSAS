"""温度相关易损性九宫格绘图工具。

用途：按 IDR、RIDR、PFA 和不同损伤状态绘制 3×3 温度对比面板。
做法：自动查找各温度工况的易损性工作簿，读取曲线并按温度统一着色。
使用：修改“用户编辑区”中的模型、温度、坐标范围和输出路径后运行本文件。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 用户编辑区：数据目录、模型、温度、损伤状态、坐标范围和输出路径
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "Output_data"
FIGURE_DIR = PROJECT_ROOT / "Paint" / "Figures"

MODEL = "PFSDF"
TEMPERATURES = [-20, -10, 0, 10, 20, 30, 40]
EDPS = ["IDR", "RIDR", "PFA"]
DS_COLUMNS = ["DS-1", "DS-2", "DS-3"]
X_LIM = (0.0, 1.5)
Y_LIM = (0.0, 1.0)
SAVE_PATH = FIGURE_DIR / f"fragility_temperature_panel_{MODEL}.png"
SHOW_FIGURE = False
CELSIUS = "\N{DEGREE CELSIUS}"

DS_LABELS = {
    "IDR": ["DS-1  (2.0%)", "DS-2  (3.0%)", "DS-3  (5.0%)"],
    "RIDR": ["DS-1  (0.2%)", "DS-2  (0.5%)", "DS-3  (1.0%)"],
    "PFA": ["DS-1  (0.5g)", "DS-2  (1.0g)", "DS-3  (1.5g)"],
}
# =============================================================================
# 用户编辑区结束
# =============================================================================


def configure_matplotlib() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "SimSun"]
    mpl.rcParams["font.sans-serif"] = ["SimSun"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["mathtext.fontset"] = "stix"


def find_fragility_workbook(model: str, temperature: int, edp: str) -> Path:
    folder = OUTPUT_DIR / f"MC8_{model}_{temperature}" / "MC8_IDA_data_frag"
    candidates = [
        path
        for path in folder.glob(f"*_{edp}.xlsx")
        if "IDA" not in path.name and "概率需求模型" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No fragility workbook found for {model}_{temperature} {edp}: {folder}")
    return candidates[0]


def read_fragility_curve(model: str, temperature: int, edp: str, ds_column: str) -> tuple[np.ndarray, np.ndarray]:
    path = find_fragility_workbook(model, temperature, edp)
    df = pd.read_excel(path, skiprows=1, header=0)
    if ds_column not in df.columns:
        raise KeyError(f"{path.name} does not contain column {ds_column!r}")

    x = df.iloc[:, 0].astype(float).to_numpy()
    y = df[ds_column].astype(float).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    return x[order], np.clip(y[order], Y_LIM[0], Y_LIM[1])


def plot_panel() -> Path:
    configure_matplotlib()

    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=min(TEMPERATURES), vmax=max(TEMPERATURES))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(14.2, 10.0),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.08, top=0.97, wspace=0.26, hspace=0.28)

    for row, edp in enumerate(EDPS):
        for col, ds_column in enumerate(DS_COLUMNS):
            ax = axes[row, col]
            for temperature in TEMPERATURES:
                x, y = read_fragility_curve(MODEL, temperature, edp, ds_column)
                ax.plot(x, y, linewidth=1.4, color=cmap(norm(temperature)))

            ax.set_xlim(*X_LIM)
            ax.set_ylim(*Y_LIM)
            ax.set_xticks(np.arange(0.0, 1.51, 0.5))
            ax.set_yticks(np.arange(0.0, 1.01, 0.2))
            ax.tick_params(axis="both", direction="in", which="both", labelsize=14)
            ax.grid(False)
            ax.text(
                0.88,
                0.10,
                DS_LABELS[edp][col],
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=14,
            )
            ax.set_xlabel(r"$Sa(T_1)$", fontsize=18, labelpad=2)
            if col == 0:
                ax.set_ylabel("Exceedance probability", fontsize=18)

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

    for row, marker in enumerate(["(a)", "(b)", "(c)"]):
        axes[row, 0].text(
            -0.24,
            0.50,
            marker,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=15,
        )

    cbar_ax = fig.add_axes([0.88, 0.23, 0.018, 0.56])
    cbar = fig.colorbar(scalar_map, cax=cbar_ax)
    cbar.set_ticks(TEMPERATURES)
    cbar.ax.tick_params(direction="in", labelsize=14)
    cbar.set_label(f"Temperature ({CELSIUS})", fontsize=18, labelpad=12)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
    return SAVE_PATH


def main() -> None:
    saved_path = plot_panel()
    print(f"[saved] {saved_path}")


if __name__ == "__main__":
    main()
