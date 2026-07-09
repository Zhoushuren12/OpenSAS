from __future__ import annotations

"""地震危险性曲线绘图与特征点导出工具。

用途：绘制 Sa—年均超越频率双对数曲线，并标出指定 50 年超越概率对应的 Sa。
做法：读取两列危险性数据，进行对数插值，同时导出 PNG、PDF 和特征点 CSV。
使用：修改“用户编辑区”中的数据文件、输出名称和概率标记后运行本文件。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_common import (
    CELSIUS,
    celsius_label,
    configure_matplotlib as apply_common_style,
    fragility_file_name,
    normalize_temperature_label,
)

apply_common_style()
# =============================================================================
# 用户编辑区：危险性数据、输出文件名、显示选项和概率标记
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "hazard_T1.880.txt"
PAINT_DIR = PROJECT_ROOT / "Paint"
OUTPUT_BASENAME = "Hazard_curve_T1.880"
SHOW_FIGURE = False

MARKER_CONFIGS = [
    {"label": "63% in 50 years", "probability": 0.63, "color": "red"},
    {"label": "10% in 50 years", "probability": 0.10, "color": "blue"},
    {"label": "2% in 50 years", "probability": 0.02, "color": "green"},
]
# =============================================================================
# 用户编辑区结束
# =============================================================================


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "STSong", "SimSun", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 1.0


def load_hazard_curve(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hazard curve file not found: {path}")

    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("灾害曲线数据必须全部为正值，才能使用双对数坐标。")

    df = pd.DataFrame(data, columns=["Sa", "lambda_M"]).sort_values("Sa").reset_index(drop=True)
    if (df["Sa"] <= 0).any() or (df["lambda_M"] <= 0).any():
        raise ValueError("灾害曲线数据必须全部为正值，才能使用双对数坐标。")

    return df


def annual_frequency_from_50yr_probability(probability: float, years: float = 50.0) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("超越概率必须位于 (0, 1) 内。")
    return -np.log(1.0 - probability) / years


def interpolate_sa_at_lambda(sa_values: np.ndarray, lambda_values: np.ndarray, target_lambda: float):
    lambda_min = float(np.min(lambda_values))
    lambda_max = float(np.max(lambda_values))
    if not lambda_min <= target_lambda <= lambda_max:
        raise ValueError(
            f"目标年均超越频率 {target_lambda:.6e} 超出数据范围 [{lambda_min:.6e}, {lambda_max:.6e}]。"
        )

    log_lambda_desc = np.log(lambda_values[::-1])
    log_sa_desc = np.log(sa_values[::-1])
    log_target_lambda = np.log(target_lambda)
    log_target_sa = np.interp(log_target_lambda, log_lambda_desc, log_sa_desc)
    return float(np.exp(log_target_sa))

def build_marker_points(df: pd.DataFrame) -> list[dict[str, float | str]]:
    sa_values = df["Sa"].to_numpy(dtype=float)
    lambda_values = df["lambda_M"].to_numpy(dtype=float)

    markers: list[dict[str, float | str]] = []
    for config in MARKER_CONFIGS:
        lambda_target = annual_frequency_from_50yr_probability(float(config["probability"]))
        sa_target = interpolate_sa_at_lambda(sa_values, lambda_values, lambda_target)
        markers.append(
            {
                "label": str(config["label"]),
                "probability": float(config["probability"]),
                "lambda_M": lambda_target,
                "Sa": sa_target,
                "color": str(config["color"]),
            }
        )
    return markers


def plot_hazard_curve(df: pd.DataFrame, marker_points: list[dict[str, float | str]]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=300)

    ax.plot(df["Sa"], df["lambda_M"], color="black", linewidth=1.6)

    handles = []
    labels = []
    for marker in marker_points:
        handle = ax.plot(
            marker["Sa"],
            marker["lambda_M"],
            linestyle="None",
            marker="o",
            markersize=8.5,
            markerfacecolor="none",
            markeredgecolor=marker["color"],
            markeredgewidth=1.4,
        )[0]
        handles.append(handle)
        labels.append(str(marker["label"]))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Sa\ (T_1)$", fontsize=25)
    ax.set_ylabel(r"$\lambda(M)$", fontsize=25)

    ax.grid(which="major", linestyle="-", linewidth=0.45, color="#b8b8b8", alpha=0.9)
    ax.grid(which="minor", linestyle=":", linewidth=0.35, color="#b8b8b8", alpha=0.9)
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, labelsize=20)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)

    ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        fontsize=15,
        handletextpad=0.6,
        borderaxespad=0.7,
    )

    ax.set_xlim(float(df["Sa"].min()) * 0.9, float(df["Sa"].max()) * 1.05)
    ax.set_ylim(10e-8, 1.0)

    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.97)
    return fig, ax


def export_marker_table(marker_points: list[dict[str, float | str]], output_csv: Path) -> None:
    table = pd.DataFrame(marker_points)
    table["p50_years"] = 1.0 - np.exp(-50.0 * table["lambda_M"].astype(float))
    table.to_csv(output_csv, index=False, encoding="utf-8-sig")


def save_outputs(fig: plt.Figure, marker_points: list[dict[str, float | str]]) -> None:
    png_path = PAINT_DIR / f"{OUTPUT_BASENAME}.png"
    pdf_path = PAINT_DIR / f"{OUTPUT_BASENAME}.pdf"
    csv_path = PAINT_DIR / f"{OUTPUT_BASENAME}_markers.csv"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    export_marker_table(marker_points, csv_path)

    print(f"PNG saved: {png_path}")
    print(f"PDF saved: {pdf_path}")
    print(f"鏍囪鐐规暟鎹凡淇濆瓨: {csv_path}")


def main() -> None:
    configure_matplotlib()
    df = load_hazard_curve(DATA_FILE)
    marker_points = build_marker_points(df)
    fig, _ = plot_hazard_curve(df, marker_points)
    save_outputs(fig, marker_points)
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
