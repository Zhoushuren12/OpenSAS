"""结构楼层响应剖面绘图工具。

用途：比较 TH/IDA 工况下 IDR、RIDR、PFA 随楼层的分布。
做法：读取统计 CSV 或单条 IDA 输出，按模型或温度整理为楼层剖面后叠加绘图。
使用：修改顶部 ``CONFIG`` 用户编辑区，或用命令行参数覆盖配置。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from plot_common import CELSIUS, OUTPUT_DIR, configure_matplotlib, normalize_temperature_label


# =============================================================================
# 用户编辑区：分析类型、响应指标、对比工况和输出选项
# =============================================================================
# analysis: 结果类型，"TH" 表示时程分析结果，"IDA" 表示 IDA 结果
# metric: 绘图指标，"IDR" 为层间位移角，"RIDR" 为残余层间位移角，"PFA" 为层加速度
# compare: 对比方式，"models" 表示对比不同模型，"temperatures" 表示对比同一模型的不同温度
CONFIG = {
    "analysis": "TH",
    "metric": "IDR",
    "compare": "temperatures",

    # TH 结果选项
    "level": None,  # 可选 "CLE"、"DBE" 、 "MCE" 或  None
    "stat": "median",  # 可选 "mean"、"median" 或 "p84"

    # IDA 结果选项
    "record": "1_1",  # MC8_IDA_data_out 下的记录文件夹，例如 "1_1"

    # 对比同一模型的不同温度：读取 Output_data/MC8_<model>_<temperature>/...
    "model": "PFSDF",
    "temperatures": ["-20","-10","0","10", "20","30","40"],

    # 对比不同模型：读取 Output_data/MC8_<model>/... 或 Output_data/MC8_<model>_<temperature>/...
    "models": ["MRF", "SMRF", "SMAPFDF", "SMABF"],
    "temperature": True,  # None 表示读取 MC8_<model>；例如 "20" 表示读取 MC8_<model>_20

    # 绘图选项
    "base_dir": OUTPUT_DIR,
    "scale": None,  # None 表示 IDR/RIDR 自动乘 100，其余情况乘 1
    "xlim": 0.5,
    "show_limits": False,  # 是否显示 RIDR 损伤状态限值线
    "drop_base": True,  # 是否删除 0 层/base 行
    "save": Path("Paint/Figures/PFSDF_response_profiles/PFSDF_IDR.png"),  # 例如 Path("Paint/Figures/response_profile.png")
    "show": False,  # True 表示弹出图窗；False 表示只保存或静默运行
}


configure_matplotlib()

# =============================================================================
# 用户编辑区结束；以下为程序实现，通常无需修改
# =============================================================================


METRIC_FILES = {
    "IDR": ("层间位移角.csv", "层间位移角.out", "IDR"),
    "RIDR": ("残余层间位移角.csv", "残余层间位移角.out", "RIDR"),
    "PFA": ("层加速度(g).csv", "层加速度(g).out", "PFA(g)"),
}

DEFAULT_MODELS = ["MRF", "SMRF", "SMAPFDF", "SMABF"]
DEFAULT_TEMPERATURES = ["-20", "0", "20", "40"]


@dataclass(frozen=True)
class Profile:
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
    data["base_dir"] = Path(data["base_dir"])
    data["analysis"] = str(data["analysis"]).upper()
    data["metric"] = str(data["metric"]).upper()
    data["compare"] = str(data["compare"]).lower()
    data["level"] = normalize_optional_level(data.get("level"))
    data["stat"] = str(data["stat"]).lower()
    data["models"] = split_csv(data.get("models"), DEFAULT_MODELS)
    data["temperatures"] = split_csv(data.get("temperatures"), DEFAULT_TEMPERATURES)
    if data.get("temperature") is not None:
        data["temperature"] = str(data["temperature"])
    if data.get("save") is not None:
        data["save"] = Path(data["save"])
    return SimpleNamespace(**data)


def normalize_optional_level(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "no", "false"}:
        return None
    return text.upper()


def effective_scale(args: SimpleNamespace) -> float:
    if args.scale is not None:
        return float(args.scale)
    return 100.0 if args.metric in {"IDR", "RIDR"} else 1.0


def validate_args(args: SimpleNamespace) -> None:
    if args.analysis not in {"TH", "IDA"}:
        raise ValueError('CONFIG["analysis"] must be "TH" or "IDA"')
    if args.metric not in METRIC_FILES:
        raise ValueError('CONFIG["metric"] must be "IDR", "RIDR", or "PFA"')
    if args.compare not in {"models", "temperatures"}:
        raise ValueError('CONFIG["compare"] must be "models" or "temperatures"')
    if args.level is not None and args.level not in {"CLE", "DBE", "MCE"}:
        raise ValueError('CONFIG["level"] must be "CLE", "DBE", "MCE", or None')
    if args.stat not in {"mean", "median", "p84"}:
        raise ValueError('CONFIG["stat"] must be "mean", "median", or "p84"')


def case_dir_name(model: str, temperature: str | None = None) -> str:
    if temperature is None or str(temperature).strip() == "":
        return f"MC8_{model}"
    return f"MC8_{model}_{temperature}"


def existing_first(candidates: Iterable[Path]) -> Path:
    checked: list[Path] = []
    for candidate in candidates:
        checked.append(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError("File not found. Checked:\n  " + "\n  ".join(str(p) for p in checked))


def th_csv_path(
    base_dir: Path,
    model: str,
    level: str | None,
    metric: str,
    *,
    temperature: str | None = None,
) -> Path:
    csv_name = METRIC_FILES[metric][0]
    case_dir = base_dir / case_dir_name(model, temperature)
    candidates: list[Path] = []
    if level is not None:
        candidates.append(case_dir / f"MC8_TH_{level}_data_out" / "结果统计" / csv_name)
    candidates.append(case_dir / "MC8_TH_data_out" / "结果统计" / csv_name)
    return existing_first(candidates)
    return base_dir / case_dir_name(model, temperature) / f"MC8_TH_{level}_data_out" / "结果统计" / csv_name


def ida_out_path(
    base_dir: Path,
    model: str,
    record: str,
    metric: str,
    *,
    temperature: str | None = None,
) -> Path:
    out_name = METRIC_FILES[metric][1]
    candidates = [base_dir / case_dir_name(model, temperature) / "MC8_IDA_data_out" / record / out_name]
    if temperature is not None:
        candidates.append(base_dir / case_dir_name(model) / "MC8_IDA_data_out" / record / out_name)
    return existing_first(candidates)


def drop_base_row(floors: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if floors.size >= 2 and abs(float(floors[0])) < 1e-12:
        return floors[1:], values[1:]
    return floors, values


def read_th_profile(
    file_path: Path,
    *,
    label: str,
    stat: str,
    scale: float,
    drop_base: bool,
) -> Profile:
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    if data.shape[1] < 2:
        raise ValueError(f"Not enough data columns: {file_path}")

    floors = data[:, 0]
    records = data[:, 1:]
    if drop_base:
        floors, records = drop_base_row(floors, records)

    if stat == "mean":
        values = np.nanmean(records, axis=1)
    elif stat == "median":
        values = np.nanmedian(records, axis=1)
    elif stat == "p84":
        values = np.nanpercentile(records, 84, axis=1)
    else:
        raise ValueError(f"Unsupported statistic: {stat}")

    return Profile(label=label, floors=floors, values=values * scale)


def read_ida_profile(file_path: Path, *, label: str, metric: str, scale: float, drop_base: bool) -> Profile:
    data = np.asarray(np.loadtxt(file_path), dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError(f"Empty data file: {file_path}")

    floors = np.arange(data.size, dtype=float)
    if drop_base:
        floors, data = drop_base_row(floors, data)
    if floors.size and floors[0] == 0:
        floors = floors + 1

    values = np.abs(data) * scale if metric != "PFA" else data * scale
    return Profile(label=label, floors=floors, values=values)


def build_profiles(args: SimpleNamespace) -> list[Profile]:
    profiles: list[Profile] = []
    scale = effective_scale(args)

    if args.compare == "models":
        for model in args.models:
            label = model if args.temperature is None else f"{model}_{args.temperature}{CELSIUS}"
            try:
                if args.analysis == "TH":
                    path = th_csv_path(args.base_dir, model, args.level, args.metric, temperature=args.temperature)
                    profiles.append(
                        read_th_profile(path, label=label, stat=args.stat, scale=scale, drop_base=args.drop_base)
                    )
                else:
                    path = ida_out_path(args.base_dir, model, args.record, args.metric, temperature=args.temperature)
                    profiles.append(
                        read_ida_profile(path, label=label, metric=args.metric, scale=scale, drop_base=args.drop_base)
                    )
            except Exception as exc:
                print(f"[skip] {label}: {exc}")
    else:
        for temperature in args.temperatures:
            label = normalize_temperature_label(f"{args.model}_{temperature}{CELSIUS}")
            try:
                if args.analysis == "TH":
                    path = th_csv_path(args.base_dir, args.model, args.level, args.metric, temperature=temperature)
                    profiles.append(
                        read_th_profile(path, label=label, stat=args.stat, scale=scale, drop_base=args.drop_base)
                    )
                else:
                    path = ida_out_path(args.base_dir, args.model, args.record, args.metric, temperature=temperature)
                    profiles.append(
                        read_ida_profile(path, label=label, metric=args.metric, scale=scale, drop_base=args.drop_base)
                    )
            except Exception as exc:
                print(f"[skip] {label}: {exc}")

    return profiles


def plot_profiles(profiles: Sequence[Profile], args: SimpleNamespace) -> None:
    if not profiles:
        print("[stop] No valid data found.")
        return

    _, _, xlabel = METRIC_FILES[args.metric]
    if args.metric in {"IDR", "RIDR"}:
        xlabel = f"{xlabel} (%)"

    fig, ax = plt.subplots(figsize=(8, 6))
    max_floor = 0
    for profile in profiles:
        ax.plot(profile.values, profile.floors, marker="o", linewidth=2, label=profile.label)
        if profile.floors.size:
            max_floor = max(max_floor, int(np.nanmax(profile.floors)))

    scale = effective_scale(args)
    if args.metric == "RIDR" and args.show_limits:
        for index, x_value in enumerate([0.002, 0.005, 0.01], start=1):
            limit_x = x_value * scale
            ax.axvline(limit_x, color="black", linestyle="--", alpha=0.65)
            ax.text(limit_x, 1.2, f"DS{index}", fontsize=12, ha="left", va="top")

    ax.set_xlabel(xlabel, fontsize=25, labelpad=10)
    ax.set_ylabel("Floor", fontsize=25, labelpad=10)
    ax.tick_params(axis="both", direction="in", which="both", labelsize=18)
    ax.grid(True, linestyle="--", alpha=0.4)
    if max_floor:
        ax.set_yticks(range(1, max_floor + 1))
        ax.set_ylim(0.75, max_floor + 0.25)
    if args.xlim is not None:
        ax.set_xlim(0, args.xlim)
    ax.legend(loc="best", fontsize=16)
    plt.tight_layout(rect=[0.01, 0.01, 0.95, 0.95])

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"[saved] {args.save}")

    if args.show:
        if not args.save:
            print("[show] Figure created. If no window appears, set CONFIG['save'] or run with --save.")
        plt.show()
    else:
        if not args.save:
            print("[done] Figure created but not saved because save=None and show=False.")
        plt.close(fig)


def args_from_cli_or_config() -> SimpleNamespace:
    parser = argparse.ArgumentParser(
        description="Plot IDR/RIDR/PFA floor profiles from the CONFIG block or optional CLI overrides."
    )
    parser.add_argument("--analysis", choices=["TH", "IDA"])
    parser.add_argument("--metric", choices=sorted(METRIC_FILES))
    parser.add_argument("--compare", choices=["models", "temperatures"])
    parser.add_argument("--level")
    parser.add_argument("--stat", choices=["mean", "median", "p84"])
    parser.add_argument("--record")
    parser.add_argument("--model")
    parser.add_argument("--models")
    parser.add_argument("--temperature")
    parser.add_argument("--temperatures")
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--xlim", type=float)
    parser.add_argument("--show-limits", action="store_true")
    parser.add_argument("--drop-base", action=argparse.BooleanOptionalAction)
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
        elif key == "show_limits":
            if value:
                config["show_limits"] = True
        else:
            config[key] = value

    args = config_namespace(config)
    validate_args(args)
    return args


def main() -> None:
    args = args_from_cli_or_config()
    profiles = build_profiles(args)
    plot_profiles(profiles, args)


if __name__ == "__main__":
    main()
