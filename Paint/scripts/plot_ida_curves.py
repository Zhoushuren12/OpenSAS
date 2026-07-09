from __future__ import annotations

"""多指标 IDA 曲线绘图工具。

用途：生成 IDR、RIDR、PFA 的 IDA 分位曲线组合图，并支持温度着色和 Sa 归一化。
做法：读取各工况 IDA 工作簿，清洗曲线后计算指定分位数并组成多面板图。
使用：在文件末部 ``build_default_config`` 的“用户编辑区”修改参数后运行本文件。
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from plot_common import (
    CELSIUS,
    celsius_label,
    configure_matplotlib as apply_common_style,
    fragility_file_name,
    normalize_temperature_label,
)

apply_common_style()
plt.rc("font", family="Times New Roman")
plt.rc("mathtext", fontset="stix")


MODEL_ALIASES = {
    "SAMBF": "SMABF",
}


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class DMPanelConfig:
    dm_name: str
    xlabel: str
    axis_scale: float = 1.0
    x_upper: Optional[float] = None
    density: int = 300
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    xticks: Optional[Sequence[float]] = None
    yticks: Optional[Sequence[float]] = None


@dataclass
class CombinedIDAConfig:
    project_root: Path
    model: str = "SMABF"
    temperatures: Sequence[Union[int, float, str]] = (-20, -10, 0, 10, 20, 30, 40)
    quantiles: Sequence[int] = (16, 50, 84)
    panels: Sequence[DMPanelConfig] = field(
        default_factory=lambda: (
            DMPanelConfig(
                dm_name="IDR",
                xlabel="IDR(%)",
                axis_scale=100.0,
                x_upper=0.10,
                density=300,
                xlim=(0.0, 10.0),
                ylim=(0.0, 2.0),
                xticks=(0, 2, 4, 6, 8, 10),
                yticks=(0.0, 0.5, 1.0, 1.5, 2.0),
            ),
            DMPanelConfig(
                dm_name="RIDR",
                xlabel="RIDR(%)",
                axis_scale=100.0,
                x_upper=0.10,
                density=300,
                xlim=(0.0, 10.0),
                ylim=(0.0, 2.0),
                xticks=(0, 2, 4, 6, 8, 10),
                yticks=(0.0, 0.5, 1.0, 1.5, 2.0),
            ),
            DMPanelConfig(
                dm_name="PFA",
                xlabel="PFA(g)",
                axis_scale=1.0,
                x_upper=2.5,
                density=300,
                xlim=(0.0, 2.5),
                ylim=(0.0, 3.0),
                xticks=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
                yticks=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
            ),
        )
    )
    sheet: Union[int, str] = 0
    collapse_limits: Dict[str, Optional[float]] = field(default_factory=dict)
    normalize_y: bool = False
    mce_spec_path: Optional[Union[str, Path]] = None
    cmap_name: str = "viridis"
    figsize: Tuple[float, float] = (10.4, 10.2)
    line_width: float = 1.2
    title: Optional[str] = None
    row_tags: Sequence[str] = ("(a)", "(b)", "(c)")
    colorbar_label: str = f"Temperature ({CELSIUS})"
    save_path: Optional[Union[str, Path]] = None
    dpi: int = 300
    show_figure: bool = True

    @property
    def ylabel(self) -> str:
        if self.normalize_y:
            return r"$Sa(T_1)/Sa_{MCE}(T_1)$"
        return r"$Sa(T_1)$"


@dataclass
class QuantileFamily:
    x: np.ndarray
    y_by_quantile: Dict[int, np.ndarray]


# ============================================================
# Data reading and preprocessing
# ============================================================

def _is_index_like(column: np.ndarray) -> bool:
    values = column[np.isfinite(column)]
    if values.size < 2:
        return False
    if not np.all(np.isclose(values, np.round(values))):
        return False

    ints = values.astype(int)
    if ints[0] not in (0, 1):
        return False
    if np.any(np.diff(ints) < 0):
        return False
    return np.all(np.isin(np.diff(ints), (0, 1)))


def _normalize_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _resolve_ida_file(
    project_root: Path,
    model: str,
    temperature: Union[int, float, str],
    dm_name: str,
) -> Path:
    normalized_model = _normalize_model_name(model)
    folder = project_root / "Output_data" / f"MC8_{normalized_model}_{temperature}" / "MC8_IDA_data_frag"
    matches = sorted(folder.glob(f"IDA*_{dm_name.upper()}.xlsx"))

    if matches:
        return matches[0]
    raise FileNotFoundError(f"未找到 {dm_name} 的 IDA 文件: {folder}")


def _read_raw_sheet(file_path: Union[str, Path], sheet: Union[int, str]) -> np.ndarray:
    df = pd.read_excel(file_path, sheet_name=sheet, header=None).apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    array = df.to_numpy(dtype=float)

    if array.shape[1] % 2 != 0 and array.shape[1] >= 3 and _is_index_like(array[:, 0]):
        warnings.warn(f"检测到索引列，已自动剔除: {file_path}", RuntimeWarning)
        array = array[:, 1:]

    if array.shape[1] % 2 != 0:
        raise ValueError(f"IDA 数据列数必须为偶数，当前为 {array.shape[1]}: {file_path}")

    return array


def _prepare_curve_points(
    x: np.ndarray,
    y: np.ndarray,
    collapse_limit: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x = x[valid].astype(float)
    y = y[valid].astype(float)

    if x.size < 2:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_unique, unique_indices = np.unique(x, return_index=True)
    y = y[unique_indices]
    x = x_unique

    if collapse_limit is not None:
        exceed = x >= float(collapse_limit)
        if np.any(exceed):
            end_index = int(np.argmax(exceed))
            x = x[: end_index + 1]
            y = y[: end_index + 1]

    if x.size < 2:
        return None

    if x[0] > 0.0 or y[0] > 0.0:
        x = np.r_[0.0, x]
        y = np.r_[0.0, y]

    return x, y


def read_ida_curves(
    file_path: Union[str, Path],
    *,
    sheet: Union[int, str] = 0,
    collapse_limit: Optional[float] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    raw_array = _read_raw_sheet(file_path, sheet)
    curves: List[Tuple[np.ndarray, np.ndarray]] = []

    for pair_index in range(raw_array.shape[1] // 2):
        result = _prepare_curve_points(
            raw_array[:, 2 * pair_index],
            raw_array[:, 2 * pair_index + 1],
            collapse_limit=collapse_limit,
        )
        if result is not None:
            curves.append(result)

    if len(curves) < 2:
        raise ValueError(f"有效 IDA 曲线数量不足 2 条: {file_path}")

    return curves


def compute_quantile_family(
    curves: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    quantiles: Iterable[int],
    density: int,
    x_upper: Optional[float] = None,
) -> QuantileFamily:
    x_min = min(np.min(x) for x, _ in curves)
    x_max = max(np.max(x) for x, _ in curves)

    if x_upper is not None:
        x_max = min(x_max, float(x_upper))
    if x_max <= x_min:
        raise ValueError(f"x_upper={x_upper} 导致统计区间无效。")

    x_grid = np.linspace(x_min, x_max, int(density))
    y_matrix = np.full((len(curves), x_grid.size), np.nan, dtype=float)

    for curve_index, (x, y) in enumerate(curves):
        valid_range = (x_grid >= x.min()) & (x_grid <= min(x.max(), x_max))
        if np.any(valid_range):
            y_matrix[curve_index, valid_range] = np.interp(x_grid[valid_range], x, y)

    available = np.any(np.isfinite(y_matrix), axis=0)
    y_by_quantile: Dict[int, np.ndarray] = {}

    for quantile in quantiles:
        y_values = np.full(x_grid.shape, np.nan, dtype=float)
        if np.any(available):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                y_values[available] = np.nanpercentile(y_matrix[:, available], quantile, axis=0)
        y_by_quantile[int(quantile)] = y_values

    return QuantileFamily(x=x_grid, y_by_quantile=y_by_quantile)


# ============================================================
# Optional normalization
# ============================================================

def _read_t1_from_model_dir(model_dir: Path) -> float:
    candidates = [
        model_dir / "MC8_PO_out" / "周期(s).out",
        model_dir / "MC8_IDA_data_out" / "周期(s).out",
    ]
    for path in candidates:
        if path.exists():
            values = np.asarray(np.loadtxt(path), dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size > 0 and values[0] > 0:
                return float(values[0])
            raise ValueError(f"周期文件内容无效: {path}")
    raise FileNotFoundError(f"未找到周期文件: {candidates[0]} 或 {candidates[1]}")


def _interp_sa_at_t1(spec_path: Union[str, Path], t1: float) -> float:
    data = np.asarray(np.loadtxt(spec_path), dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"反应谱文件格式错误: {spec_path}")

    periods = data[:, 0]
    sa_values = data[:, 1]
    valid = np.isfinite(periods) & np.isfinite(sa_values)
    periods = periods[valid]
    sa_values = sa_values[valid]

    if periods.size < 2:
        raise ValueError(f"反应谱有效点不足: {spec_path}")

    order = np.argsort(periods)
    periods = periods[order]
    sa_values = sa_values[order]

    if t1 < periods.min() or t1 > periods.max():
        raise ValueError(f"T1={t1:.4f} 超出反应谱范围 [{periods.min():.4f}, {periods.max():.4f}]")

    sa_t1 = float(np.interp(t1, periods, sa_values))
    if not np.isfinite(sa_t1) or sa_t1 <= 0:
        raise ValueError(f"SaMCE(T1) 无效: {sa_t1}")
    return sa_t1


def get_y_normalization_factor(file_path: Union[str, Path], config: CombinedIDAConfig) -> float:
    if not config.normalize_y:
        return 1.0

    spectrum_path = (
        Path(config.mce_spec_path)
        if config.mce_spec_path is not None
        else config.project_root / "Spectrum" / "MCE Level Spectrum.txt"
    )

    model_dir = Path(file_path).resolve().parent.parent
    t1 = _read_t1_from_model_dir(model_dir)
    return _interp_sa_at_t1(spectrum_path, t1)


# ============================================================
# Figure assembly
# ============================================================

def _temperature_value(temperature: Union[int, float, str]) -> float:
    return float(temperature)


def collect_quantile_data(config: CombinedIDAConfig) -> Dict[str, Dict[float, QuantileFamily]]:
    panel_data: Dict[str, Dict[float, QuantileFamily]] = {}

    for panel in config.panels:
        collapse_limit = config.collapse_limits.get(panel.dm_name)
        x_upper = panel.x_upper
        if x_upper is None and panel.xlim is not None:
            x_upper = panel.xlim[1] / panel.axis_scale

        by_temperature: Dict[float, QuantileFamily] = {}
        for temperature in config.temperatures:
            numeric_temperature = _temperature_value(temperature)
            file_path = _resolve_ida_file(config.project_root, config.model, temperature, panel.dm_name)
            curves = read_ida_curves(file_path, sheet=config.sheet, collapse_limit=collapse_limit)

            y_norm = get_y_normalization_factor(file_path, config)
            curves = [(x, y / y_norm) for x, y in curves]

            quantile_family = compute_quantile_family(
                curves,
                quantiles=config.quantiles,
                density=panel.density,
                x_upper=x_upper,
            )
            quantile_family.x = quantile_family.x * panel.axis_scale
            by_temperature[numeric_temperature] = quantile_family

        panel_data[panel.dm_name] = dict(sorted(by_temperature.items()))

    return panel_data


def _style_axis(
    ax: plt.Axes,
    panel: DMPanelConfig,
    ylabel: str,
    *,
    show_ylabel: bool,
) -> None:
    ax.tick_params(axis="both", direction="in", which="both", top=True, right=True, labelsize=18)

    if panel.xlim is not None:
        ax.set_xlim(*panel.xlim)
    else:
        ax.set_xlim(left=0.0)

    if panel.ylim is not None:
        ax.set_ylim(*panel.ylim)
    else:
        ax.set_ylim(bottom=0.0)

    if panel.xticks is not None:
        ax.set_xticks(panel.xticks)
    if panel.yticks is not None:
        ax.set_yticks(panel.yticks)

    ax.set_xlabel(panel.xlabel, fontsize=20, labelpad=8)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=20, labelpad=8)
    else:
        ax.tick_params(labelleft=False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_combined_ida_figure(config: CombinedIDAConfig) -> Tuple[plt.Figure, np.ndarray]:
    panel_data = collect_quantile_data(config)

    n_rows = len(config.panels)
    n_cols = len(config.quantiles)
    temperatures = [_temperature_value(value) for value in config.temperatures]
    cmap = plt.get_cmap(config.cmap_name)
    norm = Normalize(vmin=min(temperatures), vmax=max(temperatures))

    fig = plt.figure(figsize=config.figsize)
    grid = GridSpec(
        nrows=n_rows,
        ncols=n_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_cols + [0.18],
        hspace=0.32,
        wspace=0.28,
    )

    axes = np.empty((n_rows, n_cols), dtype=object)

    for row_index, panel in enumerate(config.panels):
        for col_index, quantile in enumerate(config.quantiles):
            ax = fig.add_subplot(grid[row_index, col_index])
            axes[row_index, col_index] = ax

            for temperature, family in panel_data[panel.dm_name].items():
                ax.plot(
                    family.x,
                    family.y_by_quantile[int(quantile)],
                    color=cmap(norm(temperature)),
                    lw=config.line_width,
                )

            _style_axis(ax, panel, config.ylabel, show_ylabel=(col_index == 0))
            ax.text(
                0.05,
                0.90,
                f"{quantile}th",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=18,
            )

            if col_index == 0 and row_index < len(config.row_tags):
                ax.text(
                    -0.15,
                    -0.16,
                    config.row_tags[row_index],
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=16,
                )

        cax = fig.add_subplot(grid[row_index, n_cols])
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.set_ticks(temperatures)
        cbar.ax.tick_params(labelsize=16, direction="in")
        cbar.set_label(config.colorbar_label, fontsize=16, rotation=0, labelpad=12)
        cbar.outline.set_linewidth(1.0)

    if config.title:
        fig.suptitle(config.title, fontsize=20, y=0.985)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.95)

    if config.save_path is not None:
        save_path = Path(config.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight")

    if config.show_figure:
        backend_name = plt.get_backend().lower()
        if "agg" in backend_name:
            plt.close(fig)
        else:
            plt.show()
    else:
        plt.close(fig)

    return fig, axes


# ============================================================
# Main entry
# ============================================================

# =============================================================================
# 用户编辑区：默认模型、温度、分位数、各面板坐标轴及输出选项
# 修改本函数 return 中的参数即可；其余函数通常无需调整。
# =============================================================================
def build_default_config() -> CombinedIDAConfig:
    project_root = Path(__file__).resolve().parents[2]
    model = "PFSDF"

    return CombinedIDAConfig(
        project_root=project_root,
        model=model,
        temperatures=(-20, -10, 0, 10, 20, 30, 40),
        quantiles=(16, 50, 84),
        panels=(
            DMPanelConfig(
                dm_name="IDR",
                xlabel="IDR(%)",
                axis_scale=100.0,
                x_upper=0.10,
                density=15,
                xlim=(0.0, 10.0),
                ylim=(0.0, 1.5),
                xticks=(0, 2, 4, 6, 8, 10),
                yticks=(0.0, 0.5, 1.0, 1.5),
            ),
            DMPanelConfig(
                dm_name="RIDR",
                xlabel="RIDR(%)",
                axis_scale=100.0,
                x_upper=0.02,
                density=25,
                xlim=(0.0, 2.0),
                ylim=(0.0, 1.5),
                xticks=(0, 0.5, 1.0, 1.5, 2.0),
                yticks=(0.0, 0.5, 1.0, 1.5),
            ),
            DMPanelConfig(
                dm_name="PFA",
                xlabel="PFA(g)",
                axis_scale=1.0,
                x_upper=2.5,
                density=50,
                xlim=(0.0, 2.5),
                ylim=(0.0, 1.5),
                xticks=(0.0,0.5,1.0,1.5,2.0,2.5),
                yticks=(0.0, 0.5, 1.0, 1.5),
            ),
        ),
        collapse_limits={},
        normalize_y=False,
        cmap_name="coolwarm",
        figsize=(10.4, 10.2),
        line_width=1.25,
        title=None,
        colorbar_label=f"Temperature ({CELSIUS})",
        save_path=project_root / "Paint" / f"IDA_Combined_{model}.png",
        show_figure=True,
    )
# =============================================================================
# 用户编辑区结束
# =============================================================================


def main() -> None:
    config = build_default_config()
    plot_combined_ida_figure(config)


if __name__ == "__main__":
    main()
