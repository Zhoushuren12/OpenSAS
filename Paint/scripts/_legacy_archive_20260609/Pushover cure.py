import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# =========================
# 1. 基本绘图设置
# =========================
plt.rc('font', family='Times New Roman')
plt.rc('mathtext', fontset='stix')


# =========================
# 2. 用户配置区
# =========================
BASE_DIR = Path('Output_data')
H_BUILDING = 35600  # mm

MODELS = ['MC8_SMABF']     
# MODELS = ['MC8_SMABF', 'MC8_PFSDF']   # 同时绘制多个模型时用这一行

TEMPS = [-20, -10, 0, 10, 20, 30, 40]
# TEMPS = [-20, 0, 20, 40]              # 也可以只画部分温度

X_LIMIT = (0, 8)
Y_LIMIT = (0, None)

SAVE_FIG = False
FIG_NAME = 'pushover_curves.png'


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
    fig_name='pushover_curves.png'
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
    fig_name : str
        图片文件名。
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
                        label = f'{temp}°C'
                    else:
                        label = f'{model}, {temp}°C'

                    ax.plot(
                        x, y,
                        label=label,
                        linewidth=2.0,
                        linestyle=line_style,
                        marker=marker
                    )

                except Exception as e:
                    print(f'[跳过] {model}_{temp}: {e}')

    ax.set_xlabel('Roof drift (%)', fontsize=14)
    ax.set_ylabel(r'$V_{\mathrm{base}} / W$', fontsize=14)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.tick_params(direction='in', labelsize=12)

    ax.grid(
        linestyle='--',
        linewidth=0.6,
        alpha=0.6,
        which='both'
    )

    ax.legend(
        loc='upper left',
        fontsize=11,
        frameon=False
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()

    if save_fig:
        fig.savefig(fig_name, dpi=600, bbox_inches='tight')
        print(f'图片已保存: {fig_name}')

    plt.show()


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
    fig_name=FIG_NAME
)