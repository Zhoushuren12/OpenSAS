import matplotlib as mpl
"""Pushover 能力曲线绘图工具。

用途：比较多个模型或温度工况的屋顶位移角—基底剪力系数曲线。
做法：读取 Pushover 目录中的位移和支座反力文件，计算 V/W 后叠加绘图。
使用：修改顶部“用户编辑区”中的模型、温度、坐标范围及输出选项后运行本文件。
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# =========================
# 1. 基本绘图设置
# =========================
from plot_common import CELSIUS, normalize_temperature_label

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
mpl.rcParams['font.sans-serif'] = ['SimSun']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'stix'


# =============================================================================
# 用户编辑区：结果目录、模型、温度、坐标范围和输出选项
# =============================================================================
BASE_DIR = Path('Output_data')
H_BUILDING = 35600  # mm

MODELS = ['MC8_PFSDF']     
# MODELS = ['MC8_SMABF', 'MC8_PFSDF']   # 同时绘制多个模型时用这一行

TEMPS = [-30,-20, -10, 0, 10, 20, 30, 40]
# TEMPS = [-20, 0, 20, 40]              # 也可以只画部分温度

X_LIMIT = (0, 8)
Y_LIMIT = (0, 0.35)
LEGEND_ROWS = 4 

SAVE_FIG = True
SHOW_FIG = True
FIG_NAME = Path('Paint/Figures/pushover_curves.png')
# =============================================================================
# 用户编辑区结束
# =============================================================================


# =========================
# 3. 读取 pushover 数据
# =========================
def read_pushover_data(folder_path, h_building=35600):
    """
    读取单个模型、单个温度工况下的 pushover 数据。

    Parameters
    ----------
    folder_path : str or Path
        Pushover 输出文件夹路径。
    h_building : float
        结构总高度，单位 mm。

    Returns
    -------
    roof_drift : ndarray
        屋顶位移角，单位 %。
    base_shear_ratio : ndarray
        基底剪力系数 Vbase / Weight。
    """

    folder = Path(folder_path)

    time_file = folder / 'Time.out'
    disp_file = folder / 'Disp9.out'

    if not time_file.exists():
        raise FileNotFoundError(f'未找到文件: {time_file}')

    if not disp_file.exists():
        raise FileNotFoundError(f'未找到文件: {disp_file}')

    time = np.loadtxt(time_file)
    n_steps = len(time)

    weight = 0.0
    base_shear = np.zeros(n_steps)

    support_files = sorted(folder.glob('Support*.out'))

    if len(support_files) == 0:
        raise FileNotFoundError(f'未找到 Support*.out 文件: {folder}')

    for support_file in support_files:
        data = np.loadtxt(support_file)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[0] != n_steps:
            raise ValueError(
                f'{support_file.name} 的步数 {data.shape[0]} 与 Time.out 的步数 {n_steps} 不一致'
            )

        # 原程序中使用 data[9, 1] 作为竖向反力估计重量
        # 这里保留你的写法，但加了安全判断
        if data.shape[0] > 9 and data.shape[1] > 1:
            weight += data[9, 1]
        else:
            raise ValueError(f'{support_file.name} 数据维度不足，无法读取 data[9, 1]')

        # 原程序中使用 -data[:, 0] 作为水平支座反力
        base_shear += -data[:, 0]

    if abs(weight) < 1e-12:
        raise ZeroDivisionError(f'结构重量读取为 0，请检查 Support*.out 文件: {folder}')

    base_shear_ratio = base_shear / weight

    roof_disp = np.loadtxt(disp_file)
    roof_drift = roof_disp * 100 / h_building

    return roof_drift, base_shear_ratio


# =========================
# 4. 绘制 pushover 曲线
# =========================
def plot_pushover_curves(
    base_dir,
    models,
    temps=None,
    h_building=35600,
    xlim=(0, 8),
    ylim=(0, None),
    save_fig=False,
    show_fig=True,
    fig_name='pushover_curves.png',
    legend_rows=2,
):
    """
    绘制一个或多个模型在不同温度下的 pushover 曲线。

    Parameters
    ----------
    base_dir : str or Path
        Output_data 所在路径。
    models : list[str]
        模型名称，例如 ['MC8_SMABF'] 或 ['MC8_SMABF', 'MC8_PFSDF']。
    temps : list[int] or None
        温度列表。如果模型没有温度后缀，例如 MC8_SMRF，可传 None。
    h_building : float
        结构总高度。
    xlim : tuple
        x 轴范围。
    ylim : tuple
        y 轴范围。
    save_fig : bool
        是否保存图片。
    show_fig : bool
        是否显示图片窗口。
    fig_name : str
        图片文件名。
    legend_rows : int
        图例排列行数，例如 2 表示温度标签分两行显示。
    """

    base_dir = Path(base_dir)

    fig, ax = plt.subplots(figsize=(8, 6))

    line_styles = ['-', '--', '-.', ':']
    markers = [None, None, None, None]

    for m_idx, model in enumerate(models):
        line_style = line_styles[m_idx % len(line_styles)]
        marker = markers[m_idx % len(markers)]

        if temps is None:
            folder_path = base_dir / model / 'MC8_PO' / 'Pushover'

            try:
                x, y = read_pushover_data(folder_path, h_building)
                ax.plot(
                    x, y,
                    label=model,
                    linewidth=2.0,
                    linestyle=line_style,
                    marker=marker
                )
            except Exception as e:
                print(f'[跳过] {model}: {e}')

        else:
            for temp in temps:
                folder_path = base_dir / f'{model}_{temp}' / 'MC8_PO' / 'Pushover'

                try:
                    x, y = read_pushover_data(folder_path, h_building)

                    if len(models) == 1:
                        label = normalize_temperature_label(f'{temp}{CELSIUS}')
                    else:
                        label = normalize_temperature_label(f'{model}, {temp}{CELSIUS}')

                    ax.plot(
                        x, y,
                        label=label,
                        linewidth=2.0,
                        linestyle=line_style,
                        marker=marker
                    )

                except Exception as e:
                    print(f'[跳过] {model}_{temp}: {e}')

    ax.set_xlabel('Roof drift (%)', fontsize=25)
    ax.set_ylabel(r'$V_{\mathrm{base}} / W$', fontsize=25)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if ylim[0] == 0 and ylim[1] is not None:
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 1e-9, 0.05))

    ax.tick_params(axis='both', direction='in', which='both', labelsize=18)

    ax.grid(
        linestyle='--',
        linewidth=0.6,
        alpha=0.6,
        which='both'
    )

    if legend_rows < 1:
        raise ValueError('legend_rows 必须为正整数')
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_ncol = max(1, int(np.ceil(len(legend_labels) / legend_rows)))
    ax.legend(
        legend_handles,
        legend_labels,
        loc='upper right',
        ncol=legend_ncol,
        fontsize=16,
        frameon=False,
        handlelength=2.1,
        handletextpad=0.55,
        columnspacing=1.25,
        labelspacing=0.45,
        borderaxespad=0.65,
    )

    plt.tight_layout(rect=[0.01, 0.01, 0.95, 0.95])

    if save_fig:
        fig_name = Path(fig_name)
        fig_name.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f'图片已保存: {fig_name}')

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


# =========================
# 5. 执行绘图
# =========================
plot_pushover_curves(
    base_dir=BASE_DIR,
    models=MODELS,
    temps=TEMPS,
    h_building=H_BUILDING,
    xlim=X_LIMIT,
    ylim=Y_LIMIT,
    save_fig=SAVE_FIG,
    show_fig=SHOW_FIG,
    fig_name=FIG_NAME,
    legend_rows=LEGEND_ROWS,
)

