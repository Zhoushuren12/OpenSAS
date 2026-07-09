"""结构响应箱线图绘制工具。

用途：按模型、温度或地震水准比较 IDR、RIDR、PFA、DCF 的统计分布。
做法：读取 TH 结果 CSV，按记录最大值或全部楼层值整理数据，再绘制分组箱线图。
使用：修改顶部 ``CONFIG`` 用户编辑区，或用命令行参数覆盖配置。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plot_common import CELSIUS, OUTPUT_DIR, configure_matplotlib, normalize_temperature_label


# =============================================================================
# 用户编辑区：响应指标、分组方式、工况、数据处理和输出选项
# =============================================================================
# metric: 绘图指标，"IDR" 为层间位移角，"RIDR" 为残余层间位移角，"PFA" 为层加速度，"DCF" 为倒塌指标
# group_by: 分组方式，"levels" 表示按 TH 工况分组，"temperatures" 表示按温度分组
# box_by: 每组内的箱体类别，通常用 "models"；当 group_by="levels" 时也可以设为 "temperatures"
CONFIG = {
    "metric": "PFA",
    "group_by": "temperatures",  # "levels" 或 "temperatures"
    "box_by": "models",  # "models" 或 "temperatures"

    # TH 工况和模型/温度
    "levels": ["CLE", "DBE", "MCE"],
    "level": "MCE",  # group_by="temperatures" 时使用
    "models": ["PFSDF"],
    "temperatures": ["-30", "-20", "-10", "0", "10", "20", "30", "40"],
    "temperature": None,  # group_by="levels" 且 box_by="models" 时可指定温度；None 读取 MC8_<model>

    # 数据整理方式
    # "max_per_record": 每条地震动取所有楼层最大值后进箱体，适合整体响应对比
    # "all_values": 所有楼层、所有地震动值都进箱体，适合看总体分布
    "data_mode": "max_per_record",
    "skip_cols": None,  # 需要排除的地震动列号，按 CSV 中响应列从 1 开始计数，例如 [2, 5]
    "drop_zero_rows": False,  # True 表示删除任意响应列含 0 的楼层行，沿用旧脚本逻辑

    # 绘图选项
    "base_dir": OUTPUT_DIR,
    "scale": None,  # None 表示 IDR/RIDR 自动乘 100，其余指标乘 1
    "ylabel": None,  # None 时根据 metric 自动生成
    "ylim": (0, 2),  # 例如 (0, 10)
    "figsize": (7, 6),
    "tick_label_size": 23,  # x、y 轴刻度文字大小
    "axis_label_size": 25,  # x、y 轴标签文字大小；当前图主要控制 ylabel
    "box_width": 0.3,  # 箱体宽度；例如 0.10 较窄、0.35 较宽，None 表示自动计算
    "box_colors": ["#EC6A6A", "#70AEEC", "#72B36A", "#C58CE8", "#E6B85C", "#6A6A6A", "#E68C8C", "#8CE6E6"],
    "show_box_labels": False,  # False 时隐藏图上方的箱体类别图例（例如 PFSDF）
    "show_mean": False,
    "show_median": True,
    "show_values": True,
    "save": Path("Paint/Figures/response_boxplot_PFA.png"),  # 例如 Path("Paint/Figures/response_boxplot.png")
    "show": True,
}


configure_matplotlib()

# =============================================================================
# 用户编辑区结束；以下为程序实现，通常无需修改
# =============================================================================


METRIC_FILES = {
    "IDR": "层间位移角.csv",
    "RIDR": "残余层间位移角.csv",
    "PFA": "层加速度(g).csv",
    "DCF": "DCF.csv",
}

YLABELS = {
    "IDR": "IDR (%)",
    "RIDR": "RIDR (%)",
    "PFA": "PFA(g)",
    "DCF": "DCF",
}

DEFAULT_MODELS = ["MRF", "SMRF", "SMAPFDF", "SMABF"]
DEFAULT_TEMPERATURES = ["-20", "0", "20", "40"]
DEFAULT_LEVELS = ["CLE", "DBE", "MCE"]


@dataclass(frozen=True)
class BoxItem:
    group_label: str
    box_label: str
    data: np.ndarray


def split_csv(value: str | Sequence[str] | None, default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if not value.strip():
            return list(default)
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def config_namespace(config: dict) -> SimpleNamespace:
    data = dict(config)
    data["metric"] = str(data["metric"]).upper()
    data["group_by"] = str(data["group_by"]).lower()
    data["box_by"] = str(data["box_by"]).lower()
    data["level"] = str(data["level"]).upper()
    data["levels"] = [level.upper() for level in split_csv(data.get("levels"), DEFAULT_LEVELS)]
    data["models"] = split_csv(data.get("models"), DEFAULT_MODELS)
    data["temperatures"] = split_csv(data.get("temperatures"), DEFAULT_TEMPERATURES)
    data["base_dir"] = Path(data["base_dir"])
    data["data_mode"] = str(data["data_mode"]).lower()
    if data.get("temperature") is not None:
        data["temperature"] = str(data["temperature"])
    if data.get("save") is not None:
        data["save"] = Path(data["save"])
    if data.get("skip_cols") is not None:
        data["skip_cols"] = [int(col) for col in data["skip_cols"]]
    if data.get("box_width") is not None:
        data["box_width"] = float(data["box_width"])
    data["tick_label_size"] = float(data["tick_label_size"])
    data["axis_label_size"] = float(data["axis_label_size"])
    return SimpleNamespace(**data)


def validate_args(args: SimpleNamespace) -> None:
    if args.metric not in METRIC_FILES:
        raise ValueError('CONFIG["metric"] must be "IDR", "RIDR", "PFA", or "DCF"')
    if args.group_by not in {"levels", "temperatures"}:
        raise ValueError('CONFIG["group_by"] must be "levels" or "temperatures"')
    if args.box_by not in {"models", "temperatures"}:
        raise ValueError('CONFIG["box_by"] must be "models" or "temperatures"')
    if args.group_by == "temperatures" and args.box_by == "temperatures":
        raise ValueError('When group_by="temperatures", CONFIG["box_by"] should usually be "models"')
    if args.data_mode not in {"max_per_record", "all_values"}:
        raise ValueError('CONFIG["data_mode"] must be "max_per_record" or "all_values"')
    if args.box_width is not None and args.box_width <= 0:
        raise ValueError('CONFIG["box_width"] must be positive or None')
    if args.tick_label_size <= 0:
        raise ValueError('CONFIG["tick_label_size"] must be positive')
    if args.axis_label_size <= 0:
        raise ValueError('CONFIG["axis_label_size"] must be positive')


def effective_scale(args: SimpleNamespace) -> float:
    if args.scale is not None:
        return float(args.scale)
    return 100.0 if args.metric in {"IDR", "RIDR"} else 1.0


def case_dir_name(model: str, temperature: str | None = None) -> str:
    if temperature is None or str(temperature).strip() == "":
        return f"MC8_{model}"
    return f"MC8_{model}_{temperature}"


def response_csv_path(
    base_dir: Path,
    model: str,
    level: str,
    metric: str,
    *,
    temperature: str | None = None,
) -> Path:
    return base_dir / case_dir_name(model, temperature) / f"MC8_TH_{level}_data_out" / "结果统计" / METRIC_FILES[metric]


def read_response_values(
    file_path: Path,
    *,
    metric: str,
    data_mode: str,
    scale: float,
    skip_cols: Sequence[int] | None = None,
    drop_zero_rows: bool = False,
) -> np.ndarray:
    with open(file_path, "r", encoding="utf-8") as file:
        first_line = file.readline()
    num_columns = len(first_line.split(","))
    if num_columns < 2:
        raise ValueError(f"Not enough columns: {file_path}")

    values = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=range(1, num_columns))
    values = np.atleast_2d(np.asarray(values, dtype=float))

    if skip_cols:
        zero_based = [col - 1 for col in skip_cols]
        values = np.delete(values, zero_based, axis=1)

    values = values[np.isfinite(values).all(axis=1)]
    if drop_zero_rows:
        values = values[~np.any(values == 0, axis=1)]
    if values.size == 0:
        raise ValueError(f"No valid nonzero data: {file_path}")

    if data_mode == "max_per_record":
        data = np.nanmax(values, axis=0)
    elif data_mode == "all_values":
        data = values.T.ravel()
    else:
        raise ValueError(f"Unsupported data_mode: {data_mode}")

    if metric in {"IDR", "RIDR"}:
        data = np.abs(data)
    return data * scale


def label_temperature(temperature: str) -> str:
    return normalize_temperature_label(f"{temperature}{CELSIUS}")


def build_box_items(args: SimpleNamespace) -> list[BoxItem]:
    items: list[BoxItem] = []
    scale = effective_scale(args)

    if args.group_by == "levels":
        group_values = args.levels
        if args.box_by == "models":
            for level in group_values:
                for model in args.models:
                    label = model if args.temperature is None else f"{model}_{label_temperature(args.temperature)}"
                    path = response_csv_path(
                        args.base_dir,
                        model,
                        level,
                        args.metric,
                        temperature=args.temperature,
                    )
                    try:
                        data = read_response_values(
                            path,
                            metric=args.metric,
                            data_mode=args.data_mode,
                            scale=scale,
                            skip_cols=args.skip_cols,
                            drop_zero_rows=args.drop_zero_rows,
                        )
                        items.append(BoxItem(level, label, data))
                    except Exception as exc:
                        print(f"[skip] {level} / {label}: {exc}")
        else:
            if len(args.models) != 1:
                raise ValueError('When box_by="temperatures", set CONFIG["models"] to one model.')
            model = args.models[0]
            for level in group_values:
                for temperature in args.temperatures:
                    label = label_temperature(temperature)
                    path = response_csv_path(args.base_dir, model, level, args.metric, temperature=temperature)
                    try:
                        data = read_response_values(
                            path,
                            metric=args.metric,
                            data_mode=args.data_mode,
                            scale=scale,
                            skip_cols=args.skip_cols,
                            drop_zero_rows=args.drop_zero_rows,
                        )
                        items.append(BoxItem(level, label, data))
                    except Exception as exc:
                        print(f"[skip] {level} / {label}: {exc}")
    else:
        for temperature in args.temperatures:
            group_label = label_temperature(temperature)
            for model in args.models:
                path = response_csv_path(args.base_dir, model, args.level, args.metric, temperature=temperature)
                try:
                    data = read_response_values(
                        path,
                        metric=args.metric,
                        data_mode=args.data_mode,
                        scale=scale,
                        skip_cols=args.skip_cols,
                        drop_zero_rows=args.drop_zero_rows,
                    )
                    items.append(BoxItem(group_label, model, data))
                except Exception as exc:
                    print(f"[skip] {group_label} / {model}: {exc}")

    return items


def plot_grouped_boxplot(items: Sequence[BoxItem], args: SimpleNamespace) -> None:
    if not items:
        print("[stop] No valid data found.")
        return

    group_labels = []
    box_labels = []
    for item in items:
        if item.group_label not in group_labels:
            group_labels.append(item.group_label)
        if item.box_label not in box_labels:
            box_labels.append(item.box_label)

    data_by_key = {(item.group_label, item.box_label): item.data for item in items}
    data = []
    labels_for_boxes = []
    for group in group_labels:
        for box_label in box_labels:
            values = data_by_key.get((group, box_label))
            if values is None:
                data.append(np.array([np.nan]))
            else:
                data.append(values)
            labels_for_boxes.append(box_label)

    boxes_per_group = len(box_labels)
    group_width = 1.0
    slot_width = group_width / max(boxes_per_group, 1)
    actual_box_width = args.box_width if args.box_width is not None else min(0.2, slot_width * 0.65)
    shrink = 0.8
    positions = []
    group_edges = []
    for group_index in range(len(group_labels)):
        group_start = group_index * group_width
        group_end = group_start + group_width
        group_edges.append((group_start, group_end))
        group_center = (group_start + group_end) / 2
        for box_index in range(boxes_per_group):
            offset = (box_index - (boxes_per_group - 1) / 2) * slot_width * shrink
            positions.append(group_center + offset)

    group_centers = [
        np.mean(positions[index * boxes_per_group : (index + 1) * boxes_per_group])
        for index in range(len(group_labels))
    ]

    fig, ax = plt.subplots(figsize=args.figsize)
    fig.patch.set_facecolor("white")

    cmap = plt.get_cmap("Set1")
    group_color_cycle = cycle(cmap.colors)
    for index, (left, right) in enumerate(group_edges):
        ax.axvspan(left, right, facecolor=next(group_color_cycle), alpha=0.05, zorder=0)

    for index in range(1, len(group_labels)):
        prev_right = positions[index * boxes_per_group - 1] + slot_width / 2
        curr_left = positions[index * boxes_per_group] - slot_width / 2
        ax.axvline((prev_right + curr_left) / 2, color="black", linewidth=1)

    boxprops = dict(linewidth=1.5, color="black")
    medianprops = dict(color="black", linewidth=2) if args.show_median else dict(color="none")
    meanprops = (
        dict(marker="s", markerfacecolor="white", markeredgecolor="black", markersize=7)
        if args.show_mean
        else None
    )
    bp = ax.boxplot(
        data,
        positions=positions,
        showmeans=args.show_mean,
        patch_artist=True,
        whis=1.5,
        boxprops=boxprops,
        medianprops=medianprops,
        meanprops=meanprops,
        widths=actual_box_width,
    )

    colors = [args.box_colors[index % len(args.box_colors)] for index in range(boxes_per_group)]
    default_colors = [colors[index % boxes_per_group] for index in range(len(data))]
    for patch, color in zip(bp["boxes"], default_colors):
        patch.set_facecolor(color)
    for flier in bp["fliers"]:
        flier.set(marker="x", color="black", markersize=7)

    finite_data = [values[np.isfinite(values)] for values in data if np.isfinite(values).any()]
    if args.ylim is not None:
        ax.set_ylim(*args.ylim)
    elif finite_data:
        ymax = max(float(np.max(values)) for values in finite_data)
        ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1)

    if args.show_values:
        ymin, ymax = ax.get_ylim()
        margin = (ymax - ymin) * 0.075
        offset = actual_box_width * 0.65
        for index, dataset in enumerate(data):
            finite = dataset[np.isfinite(dataset)]
            if finite.size == 0:
                continue
            xpos = positions[index]
            if args.show_mean:
                mean_val = float(np.mean(finite))
                yval = np.clip(mean_val, ymin + margin, ymax - margin)
                ax.text(xpos + offset, yval, f"{mean_val:.2f}", ha="left", va="center", fontsize=15, rotation=90)
            if args.show_median:
                median_val = float(np.median(finite))
                yval = np.clip(median_val, ymin + margin, ymax - margin)
                ax.text(xpos - offset, yval, f"{median_val:.2f}", ha="right", va="center", fontsize=15, rotation=90)

    if args.show_box_labels:
        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=colors[index], edgecolor="black", linewidth=1.2)
            for index in range(len(box_labels))
        ]
        ax.legend(
            legend_handles,
            box_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.16),
            ncol=min(len(box_labels), 4),
            frameon=False,
            fontsize=14,
        )

    ylabel = args.ylabel or YLABELS[args.metric]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=args.tick_label_size)
    ax.set_ylabel(ylabel, fontsize=args.axis_label_size)
    ax.tick_params(
        axis="both",
        direction="in",
        which="both",
        labelsize=args.tick_label_size,
    )
    ax.set_xlim(group_edges[0][0], group_edges[-1][1])
    ax.grid(False)
    fig.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"[saved] {args.save}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def args_from_cli_or_config() -> SimpleNamespace:
    parser = argparse.ArgumentParser(
        description="Plot TH response boxplots from the CONFIG block or optional CLI overrides."
    )
    parser.add_argument("--metric", choices=sorted(METRIC_FILES))
    parser.add_argument("--group-by", choices=["levels", "temperatures"])
    parser.add_argument("--box-by", choices=["models", "temperatures"])
    parser.add_argument("--levels")
    parser.add_argument("--level")
    parser.add_argument("--models")
    parser.add_argument("--temperatures")
    parser.add_argument("--temperature")
    parser.add_argument("--data-mode", choices=["max_per_record", "all_values"])
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--ylim", nargs=2, type=float)
    parser.add_argument("--tick-label-size", type=float)
    parser.add_argument("--axis-label-size", type=float)
    parser.add_argument("--box-width", type=float)
    parser.add_argument(
        "--show-box-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show or hide the box-category legend.",
    )
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    cli = parser.parse_args()

    config = dict(CONFIG)
    for key, value in vars(cli).items():
        if value is None:
            continue
        if key == "no_show":
            if value:
                config["show"] = False
        else:
            config[key] = tuple(value) if key == "ylim" else value

    args = config_namespace(config)
    validate_args(args)
    return args


def main() -> None:
    args = args_from_cli_or_config()
    items = build_box_items(args)
    plot_grouped_boxplot(items, args)


if __name__ == "__main__":
    main()
