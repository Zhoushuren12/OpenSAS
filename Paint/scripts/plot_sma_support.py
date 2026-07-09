"""SMA 支撑滞回曲线绘图与 GIF 生成工具。

使用方法
========
只需要修改下方“用户配置区”的两个主要选项：

1. ``ANALYSIS_TYPE``：选择结果类型
   - ``"PUSHOVER"``：读取 ``MC8_PO/Pushover``。
   - ``"TH"``：读取 ``MC8_TH_<TH_LEVEL>_data/<工况>``。
   - ``"IDA"``：读取 ``MC8_IDA_data/<工况>``。

2. ``OUTPUT_MODE``：选择输出形式
   - ``"IMAGE"``：输出一张 PNG。TH 和 IDA 分别读取 ``TH_RECORD`` 和
     ``IDA_RECORD``；Pushover 不需要工况编号。
   - ``"GIF"``：输出 GIF。TH 和 IDA 按各自的 ``*_GIF_RECORDS`` 列表
     逐帧切换工况；Pushover 按加载历程逐步展开曲线。

默认数据格式为：第 1 列是力，第 2 列是位移。程序使用 ``X_COL=1``、
``Y_COL=0``，并通过 ``Y_SCALE=1/1000`` 将力从 N 转换为 kN。

在项目根目录运行：
    python Paint/scripts/plot_sma_support.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_common import (
    PROJECT_ROOT,
    celsius_label,
    configure_matplotlib as apply_common_style,
    normalize_temperature_label,
)


apply_common_style()


# =============================================================================
# 用户配置区：通常只需要修改本区域
# =============================================================================

# ---------- 1. 选择分析结果和输出形式 ----------
# 可选："PUSHOVER"、"TH"、"IDA"
ANALYSIS_TYPE = "TH"
# 可选："IMAGE"、"GIF"
OUTPUT_MODE = "GIF"

# ---------- 2. 模型和温度 ----------
# 对应目录名 Output_data/MC8_<MODEL>_<温度>。
MODEL = "PFSDF"
TEMPERATURES = ['-20','-10','0','10','20','30','40']

# 每个温度曲线的颜色。None 表示使用 Matplotlib 默认配色。
# 列表长度必须与 TEMPERATURES 相同。
SERIES_COLORS = [None, None, None, None, None, None, None]

# ---------- 3. 不同分析类型的 SMA 文件 ----------
# 切换 ANALYSIS_TYPE 时，程序会自动选择对应文件名。
SMA_FILES = {
    "PUSHOVER": "SMA1_1.out",
    "TH": "SMA6_1.out",
    "IDA": "SMA6_1.out",
}

# ---------- 4. TH 设置 ----------
# TH_LEVEL 可选："CLE"、"DBE"、"MCE"。
TH_LEVEL = "MCE"
# OUTPUT_MODE="IMAGE" 时读取的单个 TH 工况。
TH_RECORD = "1"
# OUTPUT_MODE="GIF" 时的帧顺序；当前项目 MCE 结果通常为 1～44。
TH_GIF_RECORDS = [str(index) for index in range(1, 45)]

# ---------- 5. IDA 设置 ----------
# OUTPUT_MODE="IMAGE" 时读取的单个 IDA 工况。
IDA_RECORD = "8_1"
# OUTPUT_MODE="GIF" 时的帧顺序，请按实际存在的目录修改。
IDA_GIF_RECORDS = [f"8_{index}" for index in range(1, 81)]

# ---------- 6. 输出设置 ----------
# None：程序自动保存到 Paint/png 或 Paint/gif，并自动生成文件名。
# 也可指定绝对路径或相对项目根目录的路径，例如：
# IMAGE_SAVE_PATH = "Paint/png/my_sma_curve.png"
# GIF_SAVE_PATH = "Paint/gif/my_sma_curve.gif"
IMAGE_SAVE_PATH = None
GIF_SAVE_PATH = None

# IMAGE 模式保存后是否同时显示绘图窗口。
SHOW_IMAGE = True

# 默认不添加标题；如需标题可填写字符串。
TITLE = None

# GIF 参数。PUSHOVER_GIF_FRAMES 控制 Pushover 加载历程的动画帧数。
GIF_FPS = 5
GIF_DPI = 120
PUSHOVER_GIF_FRAMES = 120

# ---------- 7. 数据列、单位和绘图范围 ----------
X_COL = 1
Y_COL = 0
X_SCALE = 1.0
Y_SCALE = 1 / 1000
X_LABEL = "Displacement (mm)"
Y_LABEL = "Force (kN)"
X_LIM = None
Y_LIM = None
FIGSIZE = (8, 6)
LINE_WIDTH = 1.6
SHOW_GRID = True
SKIP_BAD_LINES = False

# =============================================================================
# 用户配置区结束
# =============================================================================


VALID_ANALYSES = {"PUSHOVER", "TH", "IDA"}
VALID_OUTPUT_MODES = {"IMAGE", "GIF"}


def _read_xy_data(
    file_path,
    *,
    x_col=1,
    y_col=0,
    x_scale=1.0,
    y_scale=1 / 1000,
    skip_bad_lines=False,
):
    """读取单个 SMA 文件并返回清理后的位移、力数组。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if skip_bad_lines:
        array = np.genfromtxt(path, dtype=float, invalid_raise=False)
    else:
        array = np.loadtxt(path, dtype=float)

    if array.size == 0:
        raise ValueError(f"文件为空：{path}")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] <= max(x_col, y_col):
        raise ValueError(
            f"数据列不足：{path} 只有 {array.shape[1]} 列，"
            f"但请求了第 {max(x_col, y_col) + 1} 列"
        )

    x_data = array[:, x_col] * x_scale
    y_data = array[:, y_col] * y_scale
    finite = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[finite]
    y_data = y_data[finite]
    if x_data.size == 0:
        raise ValueError(f"文件中没有有效数值：{path}")
    return x_data, y_data


def _pad_limits(lower, upper, pad=0.05):
    """在数据范围两端增加留白。"""
    if not np.isfinite(lower) or not np.isfinite(upper):
        return 0.0, 1.0
    span = upper - lower if upper != lower else (abs(lower) if lower != 0 else 1.0)
    return lower - span * pad, upper + span * pad


def _data_limits(datasets):
    """计算所有有效曲线共同使用的坐标轴范围。"""
    valid = [dataset for dataset in datasets if dataset is not None]
    if not valid:
        raise FileNotFoundError("没有找到任何可绘制的 SMA 数据。")
    x_min = min(np.min(x_data) for x_data, _ in valid)
    x_max = max(np.max(x_data) for x_data, _ in valid)
    y_min = min(np.min(y_data) for _, y_data in valid)
    y_max = max(np.max(y_data) for _, y_data in valid)
    return _pad_limits(x_min, x_max), _pad_limits(y_min, y_max)


def _style_axes(ax, *, xlabel, ylabel, grid, title=None):
    """统一设置坐标轴样式。"""
    ax.set_xlabel(normalize_temperature_label(xlabel), fontsize=25)
    ax.set_ylabel(normalize_temperature_label(ylabel), fontsize=25)
    ax.tick_params(axis="both", direction="in", which="both", labelsize=18)
    if grid:
        ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(normalize_temperature_label(title))


def plot_hysteresis_curves(
    file_paths,
    labels=None,
    *,
    colors=None,
    x_col=1,
    y_col=0,
    x_scale=1.0,
    y_scale=1 / 1000,
    title=None,
    xlabel="Displacement (mm)",
    ylabel="Force (kN)",
    figsize=(8, 6),
    grid=True,
    lw=1.6,
    xlim=None,
    ylim=None,
    save_path=None,
    show=True,
    skip_bad_lines=False,
):
    """将多个 SMA 文件绘制在一张静态对比图中。"""
    paths = [Path(path) for path in file_paths]
    labels = labels or [path.stem for path in paths]
    colors = colors or [None] * len(paths)
    if len(labels) != len(paths) or len(colors) != len(paths):
        raise ValueError("file_paths、labels 和 colors 的长度必须一致。")

    fig, ax = plt.subplots(figsize=figsize)
    plotted = 0
    first_error = None

    for path, label, color in zip(paths, labels, colors):
        try:
            x_data, y_data = _read_xy_data(
                path,
                x_col=x_col,
                y_col=y_col,
                x_scale=x_scale,
                y_scale=y_scale,
                skip_bad_lines=skip_bad_lines,
            )
        except (OSError, ValueError) as exc:
            if first_error is None:
                first_error = exc
            print(f"[警告] 无法读取，已跳过：{path}；原因：{exc}")
            continue
        ax.plot(x_data, y_data, label=label, color=color, linewidth=lw)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise FileNotFoundError(f"没有可绘制的数据。第一个错误：{first_error}")

    _style_axes(ax, xlabel=xlabel, ylabel=ylabel, grid=grid, title=title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout(rect=[0.01, 0.01, 0.95, 0.95])

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[完成] 图片已保存：{output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# 保留旧函数名，已有调用代码无需修改。
plot_hysteresis_curves_debug = plot_hysteresis_curves


def make_hysteresis_gif_from_frames(
    frame_file_groups,
    save_path,
    *,
    frame_labels=None,
    series_labels=None,
    series_colors=None,
    x_col=1,
    y_col=0,
    x_scale=1.0,
    y_scale=1 / 1000,
    title=None,
    xlabel="Displacement (mm)",
    ylabel="Force (kN)",
    figsize=(8, 6),
    grid=True,
    lw=1.6,
    xlim=None,
    ylim=None,
    fps=5,
    dpi=120,
    skip_bad_lines=False,
    show=False,
):
    """将 TH 或 IDA 的多个工况依次合成为 GIF。"""
    from matplotlib.animation import FuncAnimation

    if not frame_file_groups:
        raise ValueError("frame_file_groups 不能为空。")
    if fps <= 0:
        raise ValueError("fps 必须大于 0。")

    series_count = len(frame_file_groups[0])
    if series_count == 0:
        raise ValueError("每帧至少需要一个文件。")
    if any(len(group) != series_count for group in frame_file_groups):
        raise ValueError("每一帧包含的曲线数量必须相同。")

    frame_labels = frame_labels or [str(index) for index in range(len(frame_file_groups))]
    series_labels = series_labels or [f"Series {index + 1}" for index in range(series_count)]
    series_colors = series_colors or [None] * series_count
    if len(frame_labels) != len(frame_file_groups):
        raise ValueError("frame_labels 数量必须与帧数一致。")
    if len(series_labels) != series_count or len(series_colors) != series_count:
        raise ValueError("series_labels 和 series_colors 数量必须与每帧曲线数一致。")

    frames = []
    valid_frame_labels = []
    print(f"正在读取 {len(frame_file_groups)} 个工况……")

    for group, frame_label in zip(frame_file_groups, frame_labels):
        datasets = []
        for file_path in group:
            try:
                dataset = _read_xy_data(
                    file_path,
                    x_col=x_col,
                    y_col=y_col,
                    x_scale=x_scale,
                    y_scale=y_scale,
                    skip_bad_lines=skip_bad_lines,
                )
            except (OSError, ValueError):
                dataset = None
            datasets.append(dataset)
        if any(dataset is not None for dataset in datasets):
            frames.append(datasets)
            valid_frame_labels.append(frame_label)

    if not frames:
        first_path = Path(frame_file_groups[0][0])
        raise FileNotFoundError(
            f"没有找到可生成 GIF 的数据。第一个检查路径：{first_path}"
        )

    all_datasets = [dataset for frame in frames for dataset in frame]
    auto_xlim, auto_ylim = _data_limits(all_datasets)
    xlim = xlim or auto_xlim
    ylim = ylim or auto_ylim

    fig, ax = plt.subplots(figsize=figsize)
    _style_axes(ax, xlabel=xlabel, ylabel=ylabel, grid=grid, title=title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    lines = [
        ax.plot([], [], linewidth=lw, color=color, label=label)[0]
        for label, color in zip(series_labels, series_colors)
    ]
    ax.legend(loc="upper left")
    frame_text = ax.text(
        0.98,
        0.98,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.95, 0.95])

    def update(frame_index):
        for line, dataset in zip(lines, frames[frame_index]):
            if dataset is None:
                line.set_data([], [])
            else:
                line.set_data(*dataset)
        frame_text.set_text(f"Record: {valid_frame_labels[frame_index]}")
        return [*lines, frame_text]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=int(1000 / fps),
        blit=False,
    )
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer="pillow", fps=fps, dpi=dpi)
    print(f"[完成] GIF 已保存：{output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def make_progressive_hysteresis_gif(
    file_paths,
    save_path,
    *,
    labels=None,
    colors=None,
    frame_count=120,
    x_col=1,
    y_col=0,
    x_scale=1.0,
    y_scale=1 / 1000,
    title=None,
    xlabel="Displacement (mm)",
    ylabel="Force (kN)",
    figsize=(8, 6),
    grid=True,
    lw=1.6,
    xlim=None,
    ylim=None,
    fps=5,
    dpi=120,
    skip_bad_lines=False,
):
    """将 Pushover 文件按加载历程逐步展开为 GIF。"""
    from matplotlib.animation import FuncAnimation

    if fps <= 0:
        raise ValueError("fps 必须大于 0。")
    if frame_count < 2:
        raise ValueError("frame_count 至少为 2。")

    paths = [Path(path) for path in file_paths]
    labels = labels or [path.stem for path in paths]
    colors = colors or [None] * len(paths)
    if len(labels) != len(paths) or len(colors) != len(paths):
        raise ValueError("file_paths、labels 和 colors 的长度必须一致。")

    datasets = []
    for path in paths:
        try:
            dataset = _read_xy_data(
                path,
                x_col=x_col,
                y_col=y_col,
                x_scale=x_scale,
                y_scale=y_scale,
                skip_bad_lines=skip_bad_lines,
            )
        except (OSError, ValueError) as exc:
            print(f"[警告] 无法读取，已跳过：{path}；原因：{exc}")
            dataset = None
        datasets.append(dataset)

    auto_xlim, auto_ylim = _data_limits(datasets)
    xlim = xlim or auto_xlim
    ylim = ylim or auto_ylim
    max_points = max(len(dataset[0]) for dataset in datasets if dataset is not None)
    actual_frame_count = min(frame_count, max_points)

    fig, ax = plt.subplots(figsize=figsize)
    _style_axes(ax, xlabel=xlabel, ylabel=ylabel, grid=grid, title=title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    lines = [
        ax.plot([], [], linewidth=lw, color=color, label=label)[0]
        for label, color in zip(labels, colors)
    ]
    ax.legend(loc="upper left")
    progress_text = ax.text(
        0.98,
        0.98,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.95, 0.95])

    def update(frame_index):
        progress = (frame_index + 1) / actual_frame_count
        for line, dataset in zip(lines, datasets):
            if dataset is None:
                line.set_data([], [])
                continue
            x_data, y_data = dataset
            end = max(1, int(np.ceil(progress * len(x_data))))
            line.set_data(x_data[:end], y_data[:end])
        progress_text.set_text(f"{progress:.0%}")
        return [*lines, progress_text]

    animation = FuncAnimation(
        fig,
        update,
        frames=actual_frame_count,
        interval=int(1000 / fps),
        blit=False,
    )
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer="pillow", fps=fps, dpi=dpi)
    print(f"[完成] GIF 已保存：{output_path}")
    plt.close(fig)


def build_data_path(analysis_type, temperature, sma_file, record=None):
    """根据分析类型构造 SMA 输出文件的绝对路径。"""
    case_dir = PROJECT_ROOT / "Output_data" / f"MC8_{MODEL}_{temperature}"
    if analysis_type == "PUSHOVER":
        return case_dir / "MC8_PO" / "Pushover" / sma_file
    if analysis_type == "TH":
        if record is None:
            raise ValueError("TH 结果必须指定工况编号。")
        return case_dir / f"MC8_TH_{TH_LEVEL}_data" / str(record) / sma_file
    if analysis_type == "IDA":
        if record is None:
            raise ValueError("IDA 结果必须指定工况编号。")
        return case_dir / "MC8_IDA_data" / str(record) / sma_file
    raise ValueError(f"不支持的分析类型：{analysis_type}")


def _selected_record(analysis_type):
    if analysis_type == "TH":
        return TH_RECORD
    if analysis_type == "IDA":
        return IDA_RECORD
    return None


def _gif_records(analysis_type):
    if analysis_type == "TH":
        return TH_GIF_RECORDS
    if analysis_type == "IDA":
        return IDA_GIF_RECORDS
    return []


def _resolve_output_path(custom_path, analysis_type, output_mode, record=None):
    """使用用户路径，或按当前配置自动生成输出路径。"""
    extension = ".png" if output_mode == "IMAGE" else ".gif"
    output_folder = "png" if output_mode == "IMAGE" else "gif"
    parts = ["SMA", analysis_type]
    if analysis_type == "TH":
        parts.append(TH_LEVEL)
    parts.append(MODEL)
    if record is not None and output_mode == "IMAGE":
        parts.append(str(record))
    if output_mode == "GIF":
        parts.append("progressive" if analysis_type == "PUSHOVER" else "records")
    default_name = "_".join(parts) + extension

    if custom_path is None:
        return PROJECT_ROOT / "Paint" / output_folder / default_name
    path = Path(custom_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run():
    """读取用户配置并执行绘图任务。"""
    analysis_type = ANALYSIS_TYPE.strip().upper()
    output_mode = OUTPUT_MODE.strip().upper()
    if analysis_type not in VALID_ANALYSES:
        raise ValueError(
            f"ANALYSIS_TYPE 必须是 {sorted(VALID_ANALYSES)}，当前为 {ANALYSIS_TYPE!r}"
        )
    if output_mode not in VALID_OUTPUT_MODES:
        raise ValueError(
            f"OUTPUT_MODE 必须是 {sorted(VALID_OUTPUT_MODES)}，当前为 {OUTPUT_MODE!r}"
        )
    if len(SERIES_COLORS) != len(TEMPERATURES):
        raise ValueError("SERIES_COLORS 的长度必须与 TEMPERATURES 相同。")

    sma_file = SMA_FILES[analysis_type]
    labels = [celsius_label(temperature) for temperature in TEMPERATURES]
    record = _selected_record(analysis_type)

    if output_mode == "IMAGE":
        paths = [
            build_data_path(analysis_type, temperature, sma_file, record)
            for temperature in TEMPERATURES
        ]
        output_path = _resolve_output_path(
            IMAGE_SAVE_PATH, analysis_type, output_mode, record
        )
        plot_hysteresis_curves(
            paths,
            labels=labels,
            colors=SERIES_COLORS,
            x_col=X_COL,
            y_col=Y_COL,
            x_scale=X_SCALE,
            y_scale=Y_SCALE,
            title=TITLE,
            xlabel=X_LABEL,
            ylabel=Y_LABEL,
            figsize=FIGSIZE,
            grid=SHOW_GRID,
            lw=LINE_WIDTH,
            xlim=X_LIM,
            ylim=Y_LIM,
            save_path=output_path,
            show=SHOW_IMAGE,
            skip_bad_lines=SKIP_BAD_LINES,
        )
        return output_path

    output_path = _resolve_output_path(GIF_SAVE_PATH, analysis_type, output_mode)
    if analysis_type == "PUSHOVER":
        paths = [
            build_data_path(analysis_type, temperature, sma_file)
            for temperature in TEMPERATURES
        ]
        make_progressive_hysteresis_gif(
            paths,
            output_path,
            labels=labels,
            colors=SERIES_COLORS,
            frame_count=PUSHOVER_GIF_FRAMES,
            x_col=X_COL,
            y_col=Y_COL,
            x_scale=X_SCALE,
            y_scale=Y_SCALE,
            title=TITLE,
            xlabel=X_LABEL,
            ylabel=Y_LABEL,
            figsize=FIGSIZE,
            grid=SHOW_GRID,
            lw=LINE_WIDTH,
            xlim=X_LIM,
            ylim=Y_LIM,
            fps=GIF_FPS,
            dpi=GIF_DPI,
            skip_bad_lines=SKIP_BAD_LINES,
        )
        return output_path

    records = _gif_records(analysis_type)
    if not records:
        raise ValueError(f"{analysis_type}_GIF_RECORDS 不能为空。")
    frame_groups = [
        [
            build_data_path(analysis_type, temperature, sma_file, record_name)
            for temperature in TEMPERATURES
        ]
        for record_name in records
    ]
    make_hysteresis_gif_from_frames(
        frame_groups,
        output_path,
        frame_labels=records,
        series_labels=labels,
        series_colors=SERIES_COLORS,
        x_col=X_COL,
        y_col=Y_COL,
        x_scale=X_SCALE,
        y_scale=Y_SCALE,
        title=TITLE,
        xlabel=X_LABEL,
        ylabel=Y_LABEL,
        figsize=FIGSIZE,
        grid=SHOW_GRID,
        lw=LINE_WIDTH,
        xlim=X_LIM,
        ylim=Y_LIM,
        fps=GIF_FPS,
        dpi=GIF_DPI,
        skip_bad_lines=SKIP_BAD_LINES,
        show=False,
    )
    return output_path


if __name__ == "__main__":
    run()
