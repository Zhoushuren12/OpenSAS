import matplotlib.pyplot as plt
"""结构易损性曲线与曲面绘图工具。

用途：绘制温度相关易损性曲线、三维易损性曲面或固定 Sa 下的温度切片。
做法：读取各温度工况的易损性 Excel，按所选 EDP、损伤状态和归一化方式生成图形。
使用：只需修改文件顶部“用户编辑区”，然后直接运行本文件。
"""

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from pathlib import Path

from plot_common import (
    CELSIUS,
    celsius_label,
    configure_matplotlib as apply_common_style,
    fragility_file_name,
    normalize_temperature_label,
)

apply_common_style()
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('mathtext', fontset='stix')

# =============================================================================
# 用户编辑区：绘图类型、模型、指标、温度、坐标范围和输出选项
# =============================================================================
PLOT_TYPE = "surface"      # "curves" | "surface" | "temp_slices"
MODEL = "PFSDF"
EDP = "PFA"
DS = 2
NORMALIZE_BY_MCE = False
TEMPERATURES = [-20, -10, 0, 10, 20, 30, 40]

SA_LINES = [0.5, 1.0, 1.5]

# 横轴范围
X_LIM = (0, 1.5)
SURF_X_LIM = (0, 1.5)
SURF_T_LIM = (-20, 40)

# 曲面网格密度
NX = 301

# 配色
CMAP_NAME = "plasma"      # e.g., "plasma", "cividis", "coolwarm", "Spectral", "viridis" "RdYlBu"
SAVE_FIGURE = True
SHOW_FIGURE = False
SAVE_DPI = 300
SURFACE_DS_LABEL_FONTSIZE = 23
SURFACE_DS_LABEL_WEIGHT = 900
SURFACE_DS_LABEL_ALIGN = ('center', 'center')
SURFACE_DS_LABELS = {
    'IDR': {
        1: {'text': 'DS-1 (2.0%)', 'pos': (0.78, 0.35)},
        2: {'text': 'DS-2 (3.0%)', 'pos': (0.82, 0.35)},
        3: {'text': 'DS-3 (5.0%)', 'pos': (0.87, 0.35)},
        4: {'text': 'DS-4', 'pos': (0.87, 0.35)},
    },
    'RIDR': {
        1: {'text': 'DS-1 (0.2%)', 'pos': (0.78, 0.35)},
        2: {'text': 'DS-2 (0.5%)', 'pos': (0.82, 0.35)},
        3: {'text': 'DS-3 (1.0%)', 'pos': (0.85, 0.35)},
        4: {'text': 'DS-4', 'pos': (0.87, 0.35)},
    },
    'PFA': {
        1: {'text': 'DS-1 (0.5g)', 'pos': (0.78, 0.35)},
        2: {'text': 'DS-2 (1.0g)', 'pos': (0.80, 0.35)},
        3: {'text': 'DS-3 (1.5g)', 'pos': (0.80, 0.35)},
        4: {'text': 'DS-4', 'pos': (0.78, 0.35)},
    },
}
# =============================================================================
# 用户编辑区结束
# =============================================================================


def get_y(x: list, y: list, x0: float, error: bool = True) -> float:
    """Linear interpolation (x must be sorted)."""
    if x0 < min(x):
        if error:
            raise ValueError(f'x0 < min(x) ({x0} < {min(x)})')
        return None
    if x0 > max(x):
        if error:
            raise ValueError(f'x0 > max(x) ({x0} > {max(x)})')
        return None
    for i in range(len(x) - 1):
        if x[i] == x0:
            return y[i]
        if x[i] < x0 <= x[i + 1]:
            k = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            return k * (x0 - x[i]) + y[i]
    return y[-1]


def _get_sa_mce_for_temp(root: Path, model: str, temp: int | str):
    data_MCE = np.loadtxt(root / 'Spectrum' / 'MCE Level Spectrum.txt')
    t_path = root / 'Output_data' / f'MC8_{model}_{temp}' / 'MC8_PO_out' / '周期(s).out'
    if not t_path.exists():
        t_path = root / 'Output_data' / f'MC8_{model}' / 'MC8_PO_out' / '周期(s).out'
    T1 = float(np.loadtxt(t_path)[0])
    Sa_MCE = get_y(data_MCE[:, 0].tolist(), data_MCE[:, 1].tolist(), T1)
    return T1, Sa_MCE


def _read_fragility_df(root: Path, model: str, temp: int, edp: str) -> pd.DataFrame | None:
    path = root / 'Output_data' / f'MC8_{model}_{temp}' / 'MC8_IDA_data_frag' / f'易损性曲线_{edp}.xlsx'
    if not path.exists():
        return None
    # 你之前是 skiprows=1；按你的文件格式保留
    df = pd.read_excel(path, skiprows=1, header=0)
    return df


def _x_label() -> str:
    return r'$Sa(T_1) / Sa_{MCE}(T_1)$' if NORMALIZE_BY_MCE else r'$Sa(T_1)$'


def _surface_ds_label_config(edp: str, ds: int) -> dict:
    labels = SURFACE_DS_LABELS.get(edp.upper(), {})
    item = labels.get(ds, {'text': f'DS-{ds}', 'pos': (0.87, 0.35)})
    if isinstance(item, str):
        return {'text': item, 'pos': (0.87, 0.35)}
    return item


def _save_figure(fig: plt.Figure, root: Path, file_name: str) -> None:
    if not SAVE_FIGURE:
        return
    save_dir = root / 'Paint'
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / file_name, dpi=SAVE_DPI)


def _finalize_figure(fig: plt.Figure, root: Path, file_name: str) -> None:
    _save_figure(fig, root, file_name)
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def _extract_xy_for_temp(root: Path, model: str, temp: int, edp: str, ds: int, normalize_by_mce: bool = True):
    """
    Return x=Sa_ratio, y=exceed prob for given temp/edp/ds.
    Excel columns: 0=IM, 1=DS-1, 2=DS-2, 3=DS-3, 4=DS-4
    """
    df = _read_fragility_df(root, model, temp, edp)
    if df is None:
        return None, None

    ds_col = ds  # DS-1->1, DS-2->2, ...
    if df.shape[1] <= ds_col:
        raise ValueError(f'Excel columns insufficient for DS-{ds}: {df.shape[1]} cols in {edp}')

    _, Sa_MCE = _get_sa_mce_for_temp(root, model, temp)
    x = df.iloc[:, 0].astype(float).values
    if normalize_by_mce:
        x = x / Sa_MCE
    y = df.iloc[:, ds_col].astype(float).values

    # ????
    tmp = pd.DataFrame({'x': x, 'y': y}).groupby('x', as_index=False)['y'].max()
    x_u = tmp['x'].values
    y_u = tmp['y'].values
    order = np.argsort(x_u)
    return x_u[order], y_u[order]


def plot_fragility_curves():
    """Left figure: multiple temperature curves for one EDP & one DS (publication style)."""
    root = Path(__file__).resolve().parents[2]
    cmap = plt.get_cmap(CMAP_NAME)
    norm = mcolors.Normalize(vmin=min(TEMPERATURES), vmax=max(TEMPERATURES))

    fig, ax = plt.subplots(figsize=(8, 6))

    for temp in TEMPERATURES:
        x, y = _extract_xy_for_temp(root, MODEL, temp, EDP, DS, NORMALIZE_BY_MCE)
        if x is None:
            continue
        ax.plot(x, y * 100.0, lw=2.2, color=cmap(norm(temp)), label=celsius_label(temp))

    # 坐标轴与刻度（按你给的格式）
    ax.tick_params(axis='both', direction='in', which='both', labelsize=18)
    ax.set_xlabel(_x_label(), fontsize=25, labelpad=12)
    ax.set_ylabel('Probability of exceedance (%)', fontsize=25, labelpad=12)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(0, 100)

    # ????
    ax.legend(fontsize=18, loc='lower right',ncol=2, frameon=True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    _finalize_figure(fig, root, f'Fragility_Curves_{MODEL}_{EDP}_DS{DS}.png')

def plot_fragility_surface():
    """Middle figure: smooth joint fragility surface for one EDP & one DS (publication style)."""
    root = Path(__file__).resolve().parents[2]
    cmap = plt.get_cmap(CMAP_NAME)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    x_common = np.linspace(SURF_X_LIM[0], SURF_X_LIM[1], NX)
    surf_x_ticks = np.arange(SURF_X_LIM[0], SURF_X_LIM[1] + 0.1, 0.5)
    surf_y_ticks = np.arange(SURF_T_LIM[0], SURF_T_LIM[1] + 1, 10)

    # 收集各温度下的曲线，并插值到 x_common
    temps_used = []
    Z_list = []
    for temp in TEMPERATURES:
        x, y = _extract_xy_for_temp(root, MODEL, temp, EDP, DS, NORMALIZE_BY_MCE)
        if x is None:
            continue
        y_common = np.interp(x_common, x, y)
        temps_used.append(float(temp))
        Z_list.append(y_common)

    if not temps_used:
        raise RuntimeError("No available data to plot surface (check paths / temps).")

    temps_used = np.array(temps_used, dtype=float)
    order = np.argsort(temps_used)
    temps_used = temps_used[order]
    Z = np.array(Z_list)[order, :]  # (nT, nX)

    # 沿温度方向插值，使曲面更平滑。
    temps_dense = np.linspace(temps_used.min(), temps_used.max(), 101)
    Z_dense = np.vstack([
        np.interp(temps_dense, temps_used, Z[:, k])
        for k in range(Z.shape[1])
    ]).T  # (nTdense, nX)

    # 鐢熸垚缃戞牸
    Temp_grid = np.tile(temps_dense.reshape(-1, 1), (1, len(x_common)))
    Sa_grid = np.tile(x_common.reshape(1, -1), (len(temps_dense), 1))

    # ????
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0.01, 0.02, 0.97, 0.96], projection='3d')
    # ax = fig.add_axes(111, projection='3d')

    surf = ax.plot_surface(
        Sa_grid, Temp_grid, Z_dense,
        cmap=cmap, norm=norm,
        rcount=Z_dense.shape[0],
        ccount=Z_dense.shape[1],
        linewidth=0,
        edgecolor='none',
        antialiased=False,
        shade=False,
    )
    
    # 轴标签
    ax.set_xlabel(_x_label(), fontsize=25, labelpad=12)
    ax.set_ylabel(f'Temperature ({CELSIUS})', fontsize=25, labelpad=12)
    ax.set_zlabel('Exceedance probability', fontsize=25, labelpad=12, rotation=-90)

    # 鑼冨洿
    ax.set_xlim(*SURF_X_LIM)
    ax.set_xticks(surf_x_ticks)
    ax.set_xticklabels([''] + [f'{tick:.1f}' for tick in surf_x_ticks[1:]])
    ax.set_ylim(*SURF_T_LIM)
    ax.set_yticks(surf_y_ticks)
    ax.set_zlim(0, 1)

    ax.tick_params(axis='x', direction='in', which='both', labelsize=18, pad=2)
    ax.tick_params(axis='y', direction='in', which='both', labelsize=18, pad=2)
    ax.tick_params(axis='z', direction='in', which='both', labelsize=18, pad=4)

    # ????
    ax.view_init(elev=22, azim=-135)
    ax.text(
        SURF_X_LIM[0] + 0,
        SURF_T_LIM[0] - 5,
        0.0,
        f'{surf_x_ticks[0]:.1f}',
        fontsize=18,
        ha='center',
        va='top',
    )
    ds_label = _surface_ds_label_config(EDP, DS)
    ax.text2D(
        ds_label['pos'][0],
        ds_label['pos'][1],
        ds_label['text'],
        transform=ax.transAxes,
        fontsize=SURFACE_DS_LABEL_FONTSIZE,
        fontweight=SURFACE_DS_LABEL_WEIGHT,
        ha=SURFACE_DS_LABEL_ALIGN[0],
        va=SURFACE_DS_LABEL_ALIGN[1],
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=2.5),
    )

    # ????
    # cbar = fig.colorbar(surf, ax=ax, fraction=0.05, pad=0.08)
    # cbar.ax.tick_params(labelsize=18, direction='in')
    # cbar.set_label('超越概率(%)', fontsize=25, labelpad=12)
    # ax.set_title(f'Joint fragility surface  |  {MODEL}  {EDP}  DS-{DS}', fontsize=18, pad=10)
    # fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.90)

    _finalize_figure(fig, root, f'Fragility_Surface_{MODEL}_{EDP}_DS{DS}.png')

def plot_temp_slices():
    """Right figure: exceedance probability vs temperature at selected Sa_ratio levels (publication style)."""
    root = Path(__file__).resolve().parents[2]

    fig, ax = plt.subplots(figsize=(8, 6))

    linestyles = [
        '-', (0, (6, 3)), (0, (3, 2, 1, 2)), (0, (1, 2)), (0, (8, 2, 2, 2))
    ]

    for k, sa_ratio in enumerate(SA_LINES):
        probs = []
        temps_plot = []

        for temp in TEMPERATURES:
            x, y = _extract_xy_for_temp(root, MODEL, temp, EDP, DS, NORMALIZE_BY_MCE)
            if x is None:
                continue
            p = np.interp(sa_ratio, x, y)
            probs.append(p * 100.0)
            temps_plot.append(temp)

        if len(temps_plot) < 2:
            continue

        ax.plot(
            temps_plot, probs,
            lw=2.2,
            linestyle=linestyles[k % len(linestyles)],
            label = (rf'${sa_ratio}\times Sa_{{MCE}}$' if NORMALIZE_BY_MCE else rf'$Sa={sa_ratio}$')
        )

    # 坐标轴与刻度（按你给的格式）
    ax.set_title(f'{MODEL}  {EDP}  DS-{DS}', fontsize=18, pad=10)
    ax.tick_params(axis='both', direction='in', which='both', labelsize=18)
    ax.set_xlabel(f'Temperature ({CELSIUS})', fontsize=25, labelpad=12)
    ax.set_ylabel('Probability of exceedance (%)', fontsize=25, labelpad=12)

    ax.set_xlim(min(TEMPERATURES), max(TEMPERATURES))
    ax.set_ylim(0, 100)

    ax.legend(fontsize=18, loc='lower right', frameon=True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    _finalize_figure(fig, root, f'Fragility_Temp_Slices_{MODEL}_{EDP}_DS{DS}.png')

def main():
    if PLOT_TYPE == "curves":
        plot_fragility_curves()
    elif PLOT_TYPE == "surface":
        plot_fragility_surface()
    elif PLOT_TYPE == "temp_slices":
        plot_temp_slices()
    else:
        raise ValueError(f'Unknown PLOT_TYPE: {PLOT_TYPE}')


if __name__ == '__main__':
    main()
