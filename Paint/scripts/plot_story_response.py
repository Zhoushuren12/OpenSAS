"""楼层剪力与楼层刚度剖面绘图工具。

用途：比较 TH 或 Pushover 工况下的楼层剪力、楼层刚度分布。
做法：读取统计 CSV 或 Pushover 输出，按模型或温度构造楼层响应剖面。
使用：修改顶部 ``CONFIG`` 用户编辑区，或用命令行参数覆盖配置。
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
# 用户编辑区：响应类型、分析方式、对比工况和输出选项
# =============================================================================
# response: "shear" 表示楼层剪力，"stiffness" 表示楼层刚度
# analysis: "TH" 表示读取时程统计结果，"PO" 表示从 Pushover 输出文件计算
# compare: "models" 表示对比不同模型，"temperatures" 表示对比同一模型的不同温度
CONFIG = {
    "response": "shear",
    "analysis": "TH",
    "compare": "models",

    # TH 结果选项
    "level": "MCE",  # 可选 "CLE"、"DBE" 或 "MCE"
    "stat": "median",  # 可选 "mean"、"median" 或 "p84"

    # 对比同一模型的不同温度：读取 Output_data/MC8_<model>_<temperature>/...
    "model": "SMABF",
    "temperatures": ["-20", "0", "20", "40"],

    # 对比不同模型：读取 Output_data/MC8_<model>/... 或 Output_data/MC8_<model>_<temperature>/...
    "models": ["MRF", "SMRF", "SMAPFDF", "SMABF"],
    "temperature": None,  # None 表示读取 MC8_<model>；例如 "20" 表示读取 MC8_<model>_20

    # PO 刚度计算使用的层高，单位 mm
    "story_heights_mm": [5400, 4200, 4200, 4200, 4200, 4200, 4200, 4200],

    # 绘图选项
    "base_dir": OUTPUT_DIR,
    "xlim": None,
    "ylim": None,
    "invert_yaxis": False,
    "figsize": (8, 6),
    "save": None,  # 例如 Path("Paint/Figures/story_response.png")
    "show": True,
}


configure_matplotlib()

# =============================================================================
# 用户编辑区结束；以下为程序实现，通常无需修改
# =============================================================================


STAT_COLUMNS = {"mean": 1, "median": 4, "p84": 5}
YLABEL = "Floor"
XLABELS = {
    "shear": "Story Shear (kN)",
    "stiffness": "Story Stiffness (kN/m)",
}
DEFAULT_MODELS = ["MRF", "SMRF", "SMAPFDF", "SMABF"]
DEFAULT_TEMPERATURES = ["-20", "0", "20", "40"]


@dataclass(frozen=True)
class StoryProfile:
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
    data["response"] = str(data["response"]).lower()
    data["analysis"] = str(data["analysis"]).upper()
    data["compare"] = str(data["compare"]).lower()
    data["level"] = str(data["level"]).upper()
    data["stat"] = str(data["stat"]).lower()
    data["models"] = split_csv(data.get("models"), DEFAULT_MODELS)
    data["temperatures"] = split_csv(data.get("temperatures"), DEFAULT_TEMPERATURES)
    data["base_dir"] = Path(data["base_dir"])
    if data.get("temperature") is not None:
        data["temperature"] = str(data["temperature"])
    if data.get("save") is not None:
        data["save"] = Path(data["save"])
    return SimpleNamespace(**data)


def validate_args(args: SimpleNamespace) -> None:
    if args.response not in {"shear", "stiffness"}:
        raise ValueError('CONFIG["response"] must be "shear" or "stiffness"')
    if args.analysis not in {"TH", "PO"}:
        raise ValueError('CONFIG["analysis"] must be "TH" or "PO"')
    if args.compare not in {"models", "temperatures"}:
        raise ValueError('CONFIG["compare"] must be "models" or "temperatures"')
    if args.stat not in STAT_COLUMNS:
        raise ValueError('CONFIG["stat"] must be "mean", "median", or "p84"')


def case_dir_name(model: str, temperature: str | None = None) -> str:
    if temperature is None or str(temperature).strip() == "":
        return f"MC8_{model}"
    return f"MC8_{model}_{temperature}"


def th_stats_path(base_dir: Path, model: str, level: str, file_name: str, temperature: str | None = None) -> Path:
    return base_dir / case_dir_name(model, temperature) / f"MC8_TH_{level}_data_out" / "结果统计" / file_name


def read_stat_csv(file_path: Path, stat: str, *, drop_base: bool = False) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    floors = data[:, 0]
    values = data[:, STAT_COLUMNS[stat]]
    if drop_base and floors.size and abs(float(floors[0])) < 1e-12:
        floors = floors[1:]
        values = values[1:]
    return floors, values


def read_th_shear(args: SimpleNamespace, model: str, temperature: str | None, label: str) -> StoryProfile:
    path = th_stats_path(args.base_dir, model, args.level, "楼层剪力_统计.csv", temperature)
    floors, values = read_stat_csv(path, args.stat)
    return StoryProfile(label, floors, values)


def read_th_stiffness(args: SimpleNamespace, model: str, temperature: str | None, label: str) -> StoryProfile:
    shear_path = th_stats_path(args.base_dir, model, args.level, "楼层剪力_统计.csv", temperature)
    drift_path = th_stats_path(args.base_dir, model, args.level, "层间位移角_统计.csv", temperature)
    floors, shear = read_stat_csv(shear_path, args.stat)
    _, drift = read_stat_csv(drift_path, args.stat, drop_base=True)
    count = min(len(floors), len(drift))
    stiffness = np.divide(
        shear[:count],
        drift[:count],
        out=np.full(count, np.nan, dtype=float),
        where=np.abs(drift[:count]) > 1e-12,
    )
    return StoryProfile(label, floors[:count], stiffness)


def pushover_dir(base_dir: Path, model: str, temperature: str | None) -> Path:
    return base_dir / case_dir_name(model, temperature) / "MC8_PO" / "Pushover"


def read_po_shear(args: SimpleNamespace, model: str, temperature: str | None, label: str) -> StoryProfile:
    folder = pushover_dir(args.base_dir, model, temperature)
    values: list[float] = []
    for story in range(1, len(args.story_heights_mm) + 1):
        shear_story = None
        for file_path in folder.glob(f"Shear{story}_*.out"):
            data = np.loadtxt(file_path)
            series = np.atleast_2d(data)[:, 0] / 1000.0
            shear_story = series if shear_story is None else shear_story + series
        if shear_story is None:
            raise FileNotFoundError(f"No Shear{story}_*.out under {folder}")
        values.append(float(np.max(np.abs(shear_story))))
    floors = np.arange(1, len(values) + 1, dtype=float)
    return StoryProfile(label, floors, np.asarray(values))


def read_po_stiffness(args: SimpleNamespace, model: str, temperature: str | None, label: str) -> StoryProfile:
    folder = pushover_dir(args.base_dir, model, temperature)
    shear_profile = read_po_shear(args, model, temperature, label)
    stiffness: list[float] = []
    for story, shear, height_mm in zip(shear_profile.floors.astype(int), shear_profile.values, args.story_heights_mm):
        sdr_path = folder / f"SDR{story}.out"
        drift = np.asarray(np.loadtxt(sdr_path), dtype=float).ravel()
        max_drift = float(np.max(np.abs(drift)))
        displacement_m = max_drift * float(height_mm) / 1000.0
        stiffness.append(np.nan if displacement_m <= 1e-12 else shear / displacement_m)
    return StoryProfile(label, shear_profile.floors, np.asarray(stiffness))


def build_profiles(args: SimpleNamespace) -> list[StoryProfile]:
    profiles: list[StoryProfile] = []

    if args.compare == "models":
        cases = [(model, args.temperature, model if args.temperature is None else f"{model}_{args.temperature}{CELSIUS}") for model in args.models]
    else:
        cases = [
            (args.model, temperature, normalize_temperature_label(f"{temperature}{CELSIUS}"))
            for temperature in args.temperatures
        ]

    for model, temperature, label in cases:
        try:
            if args.analysis == "TH" and args.response == "shear":
                profiles.append(read_th_shear(args, model, temperature, label))
            elif args.analysis == "TH" and args.response == "stiffness":
                profiles.append(read_th_stiffness(args, model, temperature, label))
            elif args.analysis == "PO" and args.response == "shear":
                profiles.append(read_po_shear(args, model, temperature, label))
            else:
                profiles.append(read_po_stiffness(args, model, temperature, label))
        except Exception as exc:
            print(f"[skip] {label}: {exc}")

    return profiles


def plot_profiles(profiles: Sequence[StoryProfile], args: SimpleNamespace) -> None:
    if not profiles:
        print("[stop] No valid data found.")
        return

    fig, ax = plt.subplots(figsize=args.figsize)
    max_floor = 0
    for profile in profiles:
        ax.plot(profile.values, profile.floors, marker="o", linewidth=1.8, label=profile.label)
        max_floor = max(max_floor, int(np.nanmax(profile.floors)))

    title = f"{args.analysis} {args.response.capitalize()}"
    if args.analysis == "TH":
        title += f" {args.level} {args.stat}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(XLABELS[args.response], fontsize=20, labelpad=10)
    ax.set_ylabel(YLABEL, fontsize=20, labelpad=10)
    ax.tick_params(axis="both", direction="in", which="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_yticks(range(1, max_floor + 1))
    ax.set_ylim(*(args.ylim if args.ylim is not None else (0.75, max_floor + 0.25)))
    if args.invert_yaxis:
        ax.invert_yaxis()
    if args.xlim is not None:
        ax.set_xlim(*args.xlim if isinstance(args.xlim, (tuple, list)) else (0, args.xlim))
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


def args_from_cli_or_config() -> SimpleNamespace:
    parser = argparse.ArgumentParser(description="Plot story shear or stiffness from CONFIG or CLI overrides.")
    parser.add_argument("--response", choices=["shear", "stiffness"])
    parser.add_argument("--analysis", choices=["TH", "PO"])
    parser.add_argument("--compare", choices=["models", "temperatures"])
    parser.add_argument("--level")
    parser.add_argument("--stat", choices=["mean", "median", "p84"])
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
    profiles = build_profiles(args)
    plot_profiles(profiles, args)


if __name__ == "__main__":
    main()
