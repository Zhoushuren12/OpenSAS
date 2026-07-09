"""梁铰和柱铰响应剖面绘图工具。

用途：比较不同模型或温度下各楼层的梁铰、柱铰最大塑性转角或位置响应。
做法：读取 TH/IDA 结果中的铰响应文件，按楼层和统计量整理后绘图。
使用：修改顶部 ``CONFIG`` 用户编辑区，或通过命令行参数覆盖配置。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from plot_common import CELSIUS, OUTPUT_DIR, configure_matplotlib, normalize_temperature_label


# =============================================================================
# 用户编辑区：铰类型、分析方式、对比工况、统计量和输出选项
# =============================================================================
# hinge: "beam" 表示梁铰，"column" 表示柱铰
# plot_mode: "max_per_floor" 表示每层最大塑性转角，适合模型/温度对比；
#            "detail" 表示画构件位置明细，适合单个模型、单个温度/工况查看
# compare: "models" 表示对比不同模型，"temperatures" 表示对比同一模型的不同温度
CONFIG = {
    "hinge": "beam",
    "plot_mode": "max_per_floor",
    "compare": "models",

    # TH 结果选项
    "level": "MCE",  # 可选 "CLE"、"DBE" 或 "MCE"
    "stat": "50th",  # 可选 "16th"、"50th"、"84th"、"mean" 或 "std"

    # 对比同一模型的不同温度：读取 Output_data/MC8_<model>_<temperature>/...
    "model": "SMABF",
    "temperatures": ["-20", "0", "20", "40"],

    # 对比不同模型：读取 Output_data/MC8_<model>/... 或 Output_data/MC8_<model>_<temperature>/...
    "models": ["MRF", "SMRF", "SMAPFDF", "SMABF"],
    "temperature": None,  # None 表示读取 MC8_<model>；例如 "20" 表示读取 MC8_<model>_20

    # 绘图选项
    "base_dir": OUTPUT_DIR,
    "xlim": None,
    "ylim": None,
    "invert_yaxis": False,
    "figsize": (8, 6),
    "save": None,  # 例如 Path("Paint/Figures/hinge_profile.png")
    "show": True,
}


configure_matplotlib()

# =============================================================================
# 用户编辑区结束；以下为程序实现，通常无需修改
# =============================================================================


DEFAULT_MODELS = ["MRF", "SMRF", "SMAPFDF", "SMABF"]
DEFAULT_TEMPERATURES = ["-20", "0", "20", "40"]
VALID_STATS = {"16th", "50th", "84th", "mean", "std"}


@dataclass(frozen=True)
class HingeProfile:
    label: str
    floors: np.ndarray
    values: np.ndarray


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
    data["hinge"] = str(data["hinge"]).lower()
    data["plot_mode"] = str(data["plot_mode"]).lower()
    data["compare"] = str(data["compare"]).lower()
    data["level"] = str(data["level"]).upper()
    data["stat"] = str(data["stat"])
    data["models"] = split_csv(data.get("models"), DEFAULT_MODELS)
    data["temperatures"] = split_csv(data.get("temperatures"), DEFAULT_TEMPERATURES)
    data["base_dir"] = Path(data["base_dir"])
    if data.get("temperature") is not None:
        data["temperature"] = str(data["temperature"])
    if data.get("save") is not None:
        data["save"] = Path(data["save"])
    return SimpleNamespace(**data)


def validate_args(args: SimpleNamespace) -> None:
    if args.hinge not in {"beam", "column"}:
        raise ValueError('CONFIG["hinge"] must be "beam" or "column"')
    if args.plot_mode not in {"max_per_floor", "detail"}:
        raise ValueError('CONFIG["plot_mode"] must be "max_per_floor" or "detail"')
    if args.compare not in {"models", "temperatures"}:
        raise ValueError('CONFIG["compare"] must be "models" or "temperatures"')
    if args.stat not in VALID_STATS:
        raise ValueError('CONFIG["stat"] must be "16th", "50th", "84th", "mean", or "std"')
    if args.plot_mode == "detail":
        count = len(args.models) if args.compare == "models" else len(args.temperatures)
        if count != 1:
            raise ValueError('plot_mode="detail" requires exactly one model or one temperature.')


def case_dir_name(model: str, temperature: str | None = None) -> str:
    if temperature is None or str(temperature).strip() == "":
        return f"MC8_{model}"
    return f"MC8_{model}_{temperature}"


def hinge_file_path(
    base_dir: Path,
    model: str,
    level: str,
    hinge: str,
    stat: str,
    temperature: str | None = None,
) -> Path:
    chinese = "梁铰" if hinge == "beam" else "柱铰"
    return (
        base_dir
        / case_dir_name(model, temperature)
        / f"MC8_TH_{level}_data_out"
        / "结果统计"
        / f"{chinese}_统计_{stat}.csv"
    )


def read_numeric_csv(file_path: Path) -> tuple[list[str], np.ndarray]:
    raw = np.genfromtxt(file_path, delimiter=",", skip_header=1, dtype=str)
    raw = np.atleast_2d(raw)
    labels = [str(value) for value in raw[:, 0]]
    values = raw[:, 1:].astype(float)
    return labels, values


def floor_number(label: str) -> int:
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse floor label: {label}")
    return int(digits)


def read_max_per_floor(file_path: Path, hinge: str, label: str) -> HingeProfile:
    row_labels, values = read_numeric_csv(file_path)
    if hinge == "beam":
        floors = np.asarray([floor_number(item) for item in row_labels], dtype=float)
        max_values = np.nanmax(np.abs(values), axis=1)
        return HingeProfile(label, floors, max_values)

    floor_map: dict[int, list[np.ndarray]] = {}
    for row_label, row_values in zip(row_labels, values):
        floor_map.setdefault(floor_number(row_label), []).append(row_values)
    floors = np.asarray(sorted(floor_map), dtype=float)
    max_values = np.asarray([np.nanmax(np.abs(np.vstack(floor_map[int(floor)]))) for floor in floors])
    return HingeProfile(label, floors, max_values)


def build_cases(args: SimpleNamespace) -> list[tuple[str, str, str | None]]:
    if args.compare == "models":
        return [
            (model, model if args.temperature is None else f"{model}_{args.temperature}{CELSIUS}", args.temperature)
            for model in args.models
        ]
    return [
        (args.model, normalize_temperature_label(f"{temperature}{CELSIUS}"), temperature)
        for temperature in args.temperatures
    ]


def build_profiles(args: SimpleNamespace) -> list[HingeProfile]:
    profiles: list[HingeProfile] = []
    for model, label, temperature in build_cases(args):
        path = hinge_file_path(args.base_dir, model, args.level, args.hinge, args.stat, temperature)
        try:
            profiles.append(read_max_per_floor(path, args.hinge, label))
        except Exception as exc:
            print(f"[skip] {label}: {exc}")
    return profiles


def plot_max_profiles(profiles: Sequence[HingeProfile], args: SimpleNamespace) -> None:
    if not profiles:
        print("[stop] No valid data found.")
        return

    fig, ax = plt.subplots(figsize=args.figsize)
    max_floor = 0
    for profile in profiles:
        ax.plot(profile.values, profile.floors, marker="o", linewidth=1.8, label=profile.label)
        max_floor = max(max_floor, int(np.nanmax(profile.floors)))

    title = f"{args.level} {'Beam' if args.hinge == 'beam' else 'Column'} Hinge {args.stat}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Plastic Rotation", fontsize=20, labelpad=10)
    ax.set_ylabel("Floor", fontsize=20, labelpad=10)
    ax.tick_params(axis="both", direction="in", which="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_yticks(range(1, max_floor + 1))
    ax.set_ylim(*(args.ylim if args.ylim is not None else (0.75, max_floor + 0.25)))
    if args.invert_yaxis:
        ax.invert_yaxis()
    if args.xlim is not None:
        xlim = args.xlim if isinstance(args.xlim, (tuple, list)) else (0, args.xlim)
        ax.set_xlim(*xlim)
    ax.legend(loc="best", fontsize=12)
    fig.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"[saved] {args.save}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


def plot_detail(args: SimpleNamespace) -> None:
    model, label, temperature = build_cases(args)[0]
    path = hinge_file_path(args.base_dir, model, args.level, args.hinge, args.stat, temperature)
    row_labels, values = read_numeric_csv(path)

    fig, ax = plt.subplots(figsize=args.figsize)
    if args.hinge == "beam":
        floors = np.asarray([floor_number(item) for item in row_labels], dtype=float)
        spans = values.shape[1] // 2
        colors = plt.get_cmap("tab10").colors
        for span in range(spans):
            left = values[:, span * 2]
            right = values[:, span * 2 + 1]
            ax.plot(left, floors, marker="o", linestyle="-", color=colors[span % len(colors)], label=f"Span {span + 1} Left")
            ax.plot(right, floors, marker="s", linestyle="--", color=colors[span % len(colors)], label=f"Span {span + 1} Right")
        max_floor = int(np.nanmax(floors))
    else:
        floors = np.arange(1, len(row_labels) // 2 + 1)
        y_values = []
        for floor in floors:
            y_values.extend([floor - 0.1, floor + 0.1])
        y_values = np.asarray(y_values)
        markers = ["o", "s", "^", "d", "v", "P"]
        for col_index in range(values.shape[1]):
            ax.plot(values[:, col_index], y_values, marker=markers[col_index % len(markers)], label=f"Column {col_index + 1}")
        max_floor = int(floors.max())

    title = f"{label} {args.level} {'Beam' if args.hinge == 'beam' else 'Column'} Hinge {args.stat}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Plastic Rotation", fontsize=20, labelpad=10)
    ax.set_ylabel("Floor", fontsize=20, labelpad=10)
    ax.tick_params(axis="both", direction="in", which="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_yticks(range(1, max_floor + 1))
    ax.set_ylim(*(args.ylim if args.ylim is not None else (0.75, max_floor + 0.25)))
    if args.invert_yaxis:
        ax.invert_yaxis()
    if args.xlim is not None:
        xlim = args.xlim if isinstance(args.xlim, (tuple, list)) else (0, args.xlim)
        ax.set_xlim(*xlim)
    ax.legend(loc="best", fontsize=10)
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
    parser = argparse.ArgumentParser(description="Plot beam or column hinge profiles from CONFIG or CLI overrides.")
    parser.add_argument("--hinge", choices=["beam", "column"])
    parser.add_argument("--plot-mode", choices=["max_per_floor", "detail"])
    parser.add_argument("--compare", choices=["models", "temperatures"])
    parser.add_argument("--level")
    parser.add_argument("--stat", choices=sorted(VALID_STATS))
    parser.add_argument("--model")
    parser.add_argument("--models")
    parser.add_argument("--temperature")
    parser.add_argument("--temperatures")
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    cli = parser.parse_args()

    config = dict(CONFIG)
    for key, value in vars(cli).items():
        if value is None:
            continue
        if key == "no_show":
            config["show"] = False
        else:
            config[key] = value

    args = config_namespace(config)
    validate_args(args)
    return args


def main() -> None:
    args = args_from_cli_or_config()
    if args.plot_mode == "detail":
        plot_detail(args)
    else:
        plot_max_profiles(build_profiles(args), args)


if __name__ == "__main__":
    main()
