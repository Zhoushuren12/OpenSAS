"""Paint 绘图脚本的公共工具模块。

用途：统一项目路径、字体、坐标轴样式、温度标签、文件查找和图片保存规则。
做法：其他绘图脚本通过导入本模块复用这些常量与函数；本文件通常不单独运行。
使用：如需全局调整路径或绘图风格，只修改“用户编辑区”及对应样式函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# 用户编辑区：全局项目路径与单位
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAINT_DIR = PROJECT_ROOT / "Paint"
OUTPUT_DIR = PROJECT_ROOT / "Output_data"
SPECTRUM_DIR = PROJECT_ROOT / "Spectrum"
CELSIUS = "\N{DEGREE CELSIUS}"
# =============================================================================
# 用户编辑区结束
# =============================================================================


def configure_matplotlib() -> None:
    """Apply the shared plotting style for Paint scripts."""
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "SimSun", "Microsoft YaHei", "DejaVu Serif"]
    mpl.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["mathtext.fontset"] = "stix"


def style_axes(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel is not None:
        ax.set_xlabel(normalize_temperature_label(xlabel), fontsize=25)
    if ylabel is not None:
        ax.set_ylabel(normalize_temperature_label(ylabel), fontsize=25)
    ax.tick_params(axis="both", direction="in", which="both", labelsize=18)


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 300) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def celsius_label(value: int | float | str) -> str:
    return f"{value}{CELSIUS}"


def normalize_temperature_label(text: object) -> str:
    value = str(text)
    for old in (
        "degC",
        "DegC",
        "degree C",
        "degrees C",
        "°C",
        "掳C",
        "$^\\circ$C",
        "$\\degree C$",
        "\\degree C",
    ):
        value = value.replace(old, CELSIUS)
    return value


def fragility_file_name(edp: str) -> str:
    """Canonical fragility workbook name produced by IDA post-processing."""
    return f"易损性曲线_{edp}.xlsx"


def find_existing_file(candidates: Iterable[str | Path]) -> Path:
    checked: list[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        checked.append(path)
        if path.exists():
            return path
    joined = "\n  ".join(str(path) for path in checked)
    raise FileNotFoundError(f"未找到文件，已检查:\n  {joined}")
