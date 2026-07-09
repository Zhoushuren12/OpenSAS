"""层间位移角时程对比绘图工具。

最简单的使用流程：

1. 用 ``RESULT_TYPE`` 选择 TH 或 IDA 结果。
2. 用 ``COMPARE_MODE`` 选择比较不同温度还是不同楼层。
3. 填写 ``RECORD``，然后在项目根目录运行：
   ``python Paint/scripts/plot_time_history.py``。

脚本读取 ``Time.out``、``Disp*.out`` 或 ``SDR*.out``，可绘制单条记录，
也可批量导出多条记录。通常只需修改“用户配置区”，其余代码无需改动。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from plot_common import CELSIUS, OUTPUT_DIR, configure_matplotlib, normalize_temperature_label


# =============================================================================
# 用户配置区：按顺序修改即可
# =============================================================================

# ---------- 第 1 步：选择数据 ----------
# "TH"：普通时程结果；"IDA"：增量动力分析结果。
RESULT_TYPE = "TH"

# TH 才需要设置地震等级，可选 "CLE"、"DBE"、"MCE"；IDA 会忽略此项。
TH_LEVEL = "MCE"

# 模型名称，对应目录 MC8_<MODEL>_<温度>。
MODEL = "PFSDF"

# 要画的工况：TH 示例 "1"；IDA 示例 "2_1"。
RECORD = "1"

# ---------- 第 2 步：选择比较内容 ----------
# "TEMPERATURE"：同一楼层比较不同温度。
# "STORY"：同一温度比较不同楼层。
COMPARE_MODE = "TEMPERATURE"

# 当 COMPARE_MODE="TEMPERATURE" 时，只修改下面两项：
FIXED_STORY = 1
# TEMPERATURES = ["-20", "0", "20", "40"]
TEMPERATURES = ['-20','-10','0','10','20','30','40']

# 当 COMPARE_MODE="STORY" 时，只修改下面两项：
FIXED_TEMPERATURE = "20"
STORIES = [1, 2, 3, 4, 5, 6, 7, 8]

# ---------- 第 3 步：选择数据读取方式 ----------
# "DISPLACEMENT"：用相邻楼层 Disp 文件计算层间位移角，推荐使用。
# "SDR"：直接读取 SDR<楼层>.out。
DATA_SOURCE = "DISPLACEMENT"

# ---------- 第 4 步：保存或显示 ----------
# None 表示不保存；也可填写 "Paint/png/time_history.png"。
SAVE_PATH = None
# True 显示绘图窗口；批量导出时建议改为 False。
SHOW_FIGURE = True

# ---------- 可选：批量处理多条记录 ----------
# None：只画上面的 RECORD。
# TH 全部 44 条：[str(index) for index in range(1, 45)]
# IDA 示例：["1_1", "2_1", "3_1"]
BATCH_RECORDS = None
# 批量图片保存目录；使用 BATCH_RECORDS 时建议填写。
BATCH_SAVE_DIR = None  # 例如 "Paint/png/time_history_batch"

# ---------- 可选：坐标范围和局部放大 ----------
X_RANGE = None  # 例如 (0, 30)
Y_RANGE = None  # 例如 (-3, 3)
USE_ZOOM = False
ZOOM_TIME_RANGE = (8.0, 8.4)
ZOOM_RESPONSE_RANGE = (-1.0, 1.0)

# =============================================================================
# 用户配置区结束；以下为默认参数和程序实现，通常无需修改
# =============================================================================


def _compare_mode_value(value: object) -> str:
    """把直观的用户选项转换为程序内部名称。"""
    aliases = {
        "TEMPERATURE": "temperatures",
        "TEMPERATURES": "temperatures",
        "温度": "temperatures",
        "STORY": "stories",
        "STORIES": "stories",
        "楼层": "stories",
    }
    text = str(value).strip()
    return aliases.get(text.upper(), aliases.get(text, text.lower()))


# 将简化后的用户变量转换为原有内部配置，以兼容命令行参数和绘图函数。
CONFIG = {
    "analysis": RESULT_TYPE,
    "compare": _compare_mode_value(COMPARE_MODE),
    "level": TH_LEVEL,
    "record": RECORD,
    "records": BATCH_RECORDS,
    "batch_save_dir": BATCH_SAVE_DIR,
    "model": MODEL,
    "temperatures": TEMPERATURES,
    "story": FIXED_STORY,
    "temperature": FIXED_TEMPERATURE,
    "stories": STORIES,
    "response_source": DATA_SOURCE,
    "save": SAVE_PATH,
    "show": SHOW_FIGURE,
    "xlim": X_RANGE,
    "ylim": Y_RANGE,
    "use_inset": USE_ZOOM,
    "inset_xlim": ZOOM_TIME_RANGE,
    "inset_ylim": ZOOM_RESPONSE_RANGE,

    # 以下是统一默认值，日常使用不需要修改。
    "story_heights_mm": [5500, 4300, 4300, 4300, 4300, 4300, 4300, 4300],
    "time_file": "Time.out",
    "fallback_dt": 0.02,
    "base_dir": OUTPUT_DIR,
    "figsize": (10, 6),
    "line_width": 1.6,
    "xlabel": "Time (s)",
    "ylabel": "Inter-story drift ratio (%)",
    "title": None,
    "grid": True,
    "inset_bbox": (0.30, 0.68, 0.25, 0.25),
    "annotate_peak": True,
}


configure_matplotlib()


DEFAULT_TEMPERATURES = ["-20", "0", "20", "40"]
DEFAULT_STORIES = list(range(1, 9))


def split_csv(value: str | Sequence[str] | None, default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if not value.strip():
            return list(default)
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def split_int_csv(value: str | Sequence[int] | None, default: Sequence[int]) -> list[int]:
    return [int(item) for item in split_csv(value, [str(item) for item in default])]


def optional_limits(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    values = tuple(float(item) for item in value)
    if len(values) != 2:
        raise ValueError("Axis limits must contain exactly two values.")
    return values


def config_namespace(config: dict) -> SimpleNamespace:
    data = dict(config)
    data["analysis"] = str(data["analysis"]).upper()
    data["compare"] = str(data["compare"]).lower()
    data["level"] = str(data["level"]).upper()
    data["record"] = str(data["record"])
    records = data.get("records")
    data["records"] = None if records is None else split_csv(records, [])
    data["model"] = str(data["model"])
    data["temperatures"] = split_csv(data.get("temperatures"), DEFAULT_TEMPERATURES)
    data["temperature"] = None if data.get("temperature") is None else str(data["temperature"])
    data["story"] = int(data["story"])
    data["stories"] = split_int_csv(data.get("stories"), DEFAULT_STORIES)
    data["story_heights_mm"] = [float(value) for value in data["story_heights_mm"]]
    data["response_source"] = str(data["response_source"]).lower()
    data["base_dir"] = Path(data["base_dir"])
    data["xlim"] = optional_limits(data.get("xlim"))
    data["ylim"] = optional_limits(data.get("ylim"))
    data["inset_bbox"] = tuple(float(value) for value in data["inset_bbox"])
    data["inset_xlim"] = optional_limits(data.get("inset_xlim"))
    data["inset_ylim"] = optional_limits(data.get("inset_ylim"))
    data["save"] = None if data.get("save") is None else Path(data["save"])
    data["batch_save_dir"] = (
        None if data.get("batch_save_dir") is None else Path(data["batch_save_dir"])
    )
    return SimpleNamespace(**data)


def validate_args(args: SimpleNamespace) -> None:
    if args.analysis not in {"TH", "IDA"}:
        raise ValueError('RESULT_TYPE 只能填写 "TH" 或 "IDA"。')
    if args.compare not in {"temperatures", "stories"}:
        raise ValueError('COMPARE_MODE 只能填写 "TEMPERATURE" 或 "STORY"。')
    if args.analysis == "TH" and args.level not in {"CLE", "DBE", "MCE"}:
        raise ValueError('TH_LEVEL 只能填写 "CLE"、"DBE" 或 "MCE"。')
    if args.response_source not in {"displacement", "sdr"}:
        raise ValueError('DATA_SOURCE 只能填写 "DISPLACEMENT" 或 "SDR"。')
    if args.records is not None and not args.records:
        raise ValueError("BATCH_RECORDS 必须包含至少一个工况，或设置为 None。")
    if not args.story_heights_mm or any(height <= 0 for height in args.story_heights_mm):
        raise ValueError("楼层高度必须全部大于 0。")
    max_story = len(args.story_heights_mm)
    selected_stories = [args.story] if args.compare == "temperatures" else args.stories
    invalid = [story for story in selected_stories if not 1 <= story <= max_story]
    if invalid:
        raise ValueError(f"楼层编号必须在 1～{max_story} 之间：{invalid}")
    if args.compare == "stories" and args.temperature is None:
        raise ValueError('比较楼层时必须填写 FIXED_TEMPERATURE。')
    if args.use_inset and (args.inset_xlim is None or args.inset_ylim is None):
        raise ValueError("启用 USE_ZOOM 时必须填写两个局部放大范围。")


def case_dir_name(model: str, temperature: str | None) -> str:
    if temperature is None or not str(temperature).strip():
        return f"MC8_{model}"
    return f"MC8_{model}_{temperature}"


def analysis_case_dir(args: SimpleNamespace, temperature: str | None) -> Path:
    model_dir = args.base_dir / case_dir_name(args.model, temperature)
    if args.analysis == "TH":
        analysis_dir = model_dir / f"MC8_TH_{args.level}_data"
    else:
        analysis_dir = model_dir / "MC8_IDA_data"
    return analysis_dir / args.record


def story_file_names(story: int) -> tuple[str, str, str]:
    """Return top displacement, bottom displacement and SDR files for a 1-based story."""
    return f"Disp{story + 1}.out", f"Disp{story}.out", f"SDR{story}.out"


def build_plot_inputs(
    args: SimpleNamespace,
) -> tuple[list[Path], list[str], list[int]]:
    if args.compare == "temperatures":
        case_dirs = [analysis_case_dir(args, temperature) for temperature in args.temperatures]
        labels = [normalize_temperature_label(f"{temperature}{CELSIUS}") for temperature in args.temperatures]
        stories = [args.story] * len(case_dirs)
    else:
        case_dir = analysis_case_dir(args, args.temperature)
        case_dirs = [case_dir] * len(args.stories)
        labels = [f"Story {story}" for story in args.stories]
        stories = list(args.stories)
    return case_dirs, labels, stories


def load_vector(path: Path, column: int | None = None) -> np.ndarray:
    values = np.asarray(np.loadtxt(path), dtype=float)
    if values.ndim == 2:
        values = values[:, -1 if column is None else column]
    return values.reshape(-1)


def read_idr_history(
    case_dir: Path,
    story: int,
    args: SimpleNamespace,
) -> tuple[np.ndarray, np.ndarray]:
    top_name, bottom_name, sdr_name = story_file_names(story)
    if args.response_source == "sdr":
        response_path = case_dir / sdr_name
        if not response_path.exists():
            raise FileNotFoundError(response_path)
        idr = load_vector(response_path) * 100.0
    else:
        top_path = case_dir / top_name
        bottom_path = case_dir / bottom_name
        if not top_path.exists():
            raise FileNotFoundError(top_path)
        if not bottom_path.exists():
            raise FileNotFoundError(bottom_path)
        top_disp = load_vector(top_path)
        bottom_disp = load_vector(bottom_path)
        count = min(len(top_disp), len(bottom_disp))
        height = args.story_heights_mm[story - 1]
        idr = (top_disp[:count] - bottom_disp[:count]) / height * 100.0

    time_path = case_dir / args.time_file
    if time_path.exists() and time_path.stat().st_size > 0:
        time = load_vector(time_path, column=0)
    else:
        time = np.arange(len(idr), dtype=float) * float(args.fallback_dt)

    count = min(len(time), len(idr))
    time = time[:count]
    idr = idr[:count]
    valid = np.isfinite(time) & np.isfinite(idr)
    if not np.any(valid):
        raise ValueError(f"No valid time-history values: {case_dir}")
    return time[valid], idr[valid]


def plot_idr_timehistory(
    case_dirs: Sequence[Path],
    labels: Sequence[str],
    stories: Sequence[int],
    args: SimpleNamespace,
) -> Path | None:
    fig, ax = plt.subplots(figsize=args.figsize)
    try:
        fig.canvas.manager.set_window_title(
            f"{args.analysis} time history - record {args.record}"
        )
    except AttributeError:
        pass
    inset_ax = None
    if args.use_inset:
        inset_ax = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=args.inset_bbox,
            bbox_transform=ax.transAxes,
            loc="lower left",
        )

    plotted = 0
    for case_dir, label, story in zip(case_dirs, labels, stories):
        try:
            time, idr = read_idr_history(case_dir, story, args)
        except Exception as exc:
            print(f"[skip] {label}: {exc}")
            continue

        line = ax.plot(time, idr, linewidth=args.line_width, label=label)[0]
        plotted += 1
        if inset_ax is not None:
            inset_ax.plot(time, idr, linewidth=1.0, color=line.get_color())
            if args.annotate_peak:
                peak_index = int(np.argmax(np.abs(idr)))
                inset_ax.annotate(
                    f"{idr[peak_index]:.2f}%",
                    xy=(time[peak_index], idr[peak_index]),
                    xytext=(5, 8),
                    textcoords="offset points",
                    fontsize=11,
                    color=line.get_color(),
                    arrowprops=dict(arrowstyle="->", color=line.get_color(), linewidth=0.8),
                )

    if plotted == 0:
        plt.close(fig)
        print("[stop] No valid time-history data found.")
        return None

    if args.title:
        ax.set_title(str(args.title), fontsize=18)
    ax.set_xlabel(str(args.xlabel), fontsize=25)
    ax.set_ylabel(str(args.ylabel), fontsize=25)
    ax.tick_params(axis="both", direction="in", which="both", labelsize=18)
    if args.grid:
        ax.grid(True, linestyle="--", alpha=0.3)
    if args.xlim is not None:
        ax.set_xlim(*args.xlim)
    if args.ylim is not None:
        ax.set_ylim(*args.ylim)
    ax.legend(fontsize=14, frameon=False)

    if inset_ax is not None:
        inset_ax.set_xlim(*args.inset_xlim)
        inset_ax.set_ylim(*args.inset_ylim)
        inset_ax.tick_params(axis="both", direction="in", which="both", labelsize=10)
        mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.5")

    fig.tight_layout(rect=[0.01, 0.01, 0.95, 0.95])
    output_path = None
    if args.save is not None:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[saved] {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def args_from_cli_or_config() -> SimpleNamespace:
    parser = argparse.ArgumentParser(description="Plot TH or IDA inter-story drift-ratio time histories.")
    parser.add_argument("--analysis", choices=["TH", "IDA"])
    parser.add_argument("--compare", choices=["temperatures", "stories"])
    parser.add_argument("--level", choices=["CLE", "DBE", "MCE"])
    parser.add_argument("--record")
    parser.add_argument("--records", help="Comma-separated records, e.g. 1,2,3 or 1_1,2_1")
    parser.add_argument("--batch-save-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--temperatures", help="Comma-separated temperatures, e.g. -20,0,20,40")
    parser.add_argument("--temperature")
    parser.add_argument("--story", type=int)
    parser.add_argument("--stories", help="Comma-separated stories, e.g. 1,2,3")
    parser.add_argument("--response-source", choices=["displacement", "sdr"])
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--ylim", nargs=2, type=float)
    parser.add_argument("--use-inset", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    cli = parser.parse_args()

    config = dict(CONFIG)
    for key, value in vars(cli).items():
        if key == "no_show":
            if value:
                config["show"] = False
            continue
        if value is None:
            continue
        config[key] = tuple(value) if key in {"xlim", "ylim"} else value

    args = config_namespace(config)
    validate_args(args)
    return args


def main() -> None:
    args = args_from_cli_or_config()
    records = args.records if args.records is not None else [args.record]
    multiple_records = len(records) > 1

    for index, record in enumerate(records, start=1):
        record_args = SimpleNamespace(**vars(args))
        record_args.record = str(record)

        if args.batch_save_dir is not None:
            safe_record = "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in record_args.record
            )
            level_part = f"_{args.level.lower()}" if args.analysis == "TH" else ""
            record_args.save = (
                args.batch_save_dir
                / f"time_history_{args.analysis.lower()}{level_part}_record_{safe_record}.png"
            )
        elif multiple_records and args.save is not None:
            suffix = args.save.suffix or ".png"
            safe_record = "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in record_args.record
            )
            record_args.save = args.save.with_name(
                f"{args.save.stem}_record_{safe_record}{suffix}"
            )

        print(f"[record {index}/{len(records)}] {record_args.record}")
        case_dirs, labels, stories = build_plot_inputs(record_args)
        plot_idr_timehistory(case_dirs, labels, stories, record_args)


if __name__ == "__main__":
    main()
