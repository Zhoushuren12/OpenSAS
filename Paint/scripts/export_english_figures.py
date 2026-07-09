"""中文版论文图批量重新生成工具。

用途：从原始分析结果重新生成中文版 Pushover、易损性和区域分析图件。
做法：读取 ``Output_data`` 和区域温度数据，使用宋体中文标签绘图并输出到 ``Paint/Chinese``。
使用：修改“用户编辑区”中的模型、温度和任务开关，然后直接运行本文件。

说明：文件名因兼容既有调用仍保留为 ``export_english_figures.py``，但本文件现在只生成中文图。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_common import CELSIUS, celsius_label


# =============================================================================
# 用户编辑区：模型、温度、中文图输出目录和批量任务开关
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "Paint" / "Chinese"

MODELS = ("PFSDF", "SMABF")
SURFACE_MODELS = ("PFSDF",)
TEMPERATURES = (-20, -10, 0, 10, 20, 30, 40)
REGIONAL_HAZARD_EDPS = ("IDR", "RIDR", "PFA")

GENERATE_PUSHOVER = True
GENERATE_FRAGILITY_PANEL = True
GENERATE_FRAGILITY_SURFACE = True
GENERATE_REGIONAL_FRAGILITY = True
GENERATE_REGIONAL_HAZARD = True
# =============================================================================
# 用户编辑区结束
# =============================================================================


def setup_chinese_style() -> None:
    """Configure publication-style Chinese labels with English/math fallbacks."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "SimSun", "Microsoft YaHei"]
    plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["text.usetex"] = False


def load_script_module(script_name: str, module_name: str) -> ModuleType:
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"未找到绘图脚本: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载绘图脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_task(name: str, function: Callable[..., object], *args: object) -> None:
    try:
        function(*args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[跳过] {name}: {exc}")


def read_pushover_data(folder_path: Path, h_building: float = 35600.0) -> tuple[np.ndarray, np.ndarray]:
    time = np.loadtxt(folder_path / "Time.out")
    weight = 0.0
    shear = np.zeros(len(time))
    for index in range(1, 1000):
        support_file = folder_path / f"Support{index}.out"
        if not support_file.exists():
            continue
        data = np.loadtxt(support_file)
        weight += float(data[9, 1])
        shear += -data[:, 0]
    if np.isclose(weight, 0.0):
        raise ValueError(f"未能从支座反力文件计算结构重量: {folder_path}")
    roof_disp = np.loadtxt(folder_path / "Disp9.out")
    return roof_disp * 100.0 / h_building, shear / weight


def plot_pushover(model: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    for temperature in TEMPERATURES:
        folder = PROJECT_DIR / "Output_data" / f"MC8_{model}_{temperature}" / "MC8_PO" / "Pushover"
        if not folder.exists():
            continue
        x, y = read_pushover_data(folder)
        ax.plot(x, y, linewidth=2.2, label=celsius_label(temperature))
        plotted = True

    if not plotted:
        plt.close(fig)
        raise FileNotFoundError(f"未找到 {model} 的 Pushover 结果。")

    ax.set_xlabel("屋顶位移角 (%)", fontsize=25, fontname="SimSun", labelpad=10)
    ax.set_ylabel("基底剪力系数 ($V/W$)", fontsize=25, fontname="SimSun", labelpad=10)
    ax.tick_params(axis="both", direction="in", which="both", labelsize=18)
    ax.legend(loc="upper right", fontsize=15, frameon=False)
    ax.grid(linestyle="--", which="both", alpha=0.45)
    ax.set_xlim(0, 8)
    ax.set_ylim(bottom=0)
    fig.tight_layout(rect=[0.01, 0.01, 0.95, 0.95])
    output_path = OUT_DIR / f"Pushover曲线_{model}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[生成] {output_path}")


def find_fragility_file(model: str, temperature: int, edp: str) -> Path | None:
    folder = PROJECT_DIR / "Output_data" / f"MC8_{model}_{temperature}" / "MC8_IDA_data_frag"
    if not folder.exists():
        return None
    candidates = [
        path
        for path in folder.glob(f"*_{edp}.xlsx")
        if "IDA" not in path.name and "概率需求模型" not in path.name
    ]
    return candidates[0] if candidates else None


def plot_fragility_panel(model: str) -> None:
    ds_labels = {
        "IDR": ["DS-1 (2.0%)", "DS-2 (3.0%)", "DS-3 (5.0%)"],
        "RIDR": ["DS-1 (0.2%)", "DS-2 (0.5%)", "DS-3 (1.0%)"],
        "PFA": ["DS-1 (0.5g)", "DS-2 (1.0g)", "DS-3 (1.5g)"],
    }
    edps = list(ds_labels)
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=min(TEMPERATURES), vmax=max(TEMPERATURES))
    scalar_map = cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array([])

    fig, axes = plt.subplots(3, 3, figsize=(14.2, 10.0), sharex=True, sharey=True)
    plotted = False
    for row, edp in enumerate(edps):
        for column in range(3):
            ax = axes[row, column]
            for temperature in TEMPERATURES:
                path = find_fragility_file(model, temperature, edp)
                if path is None:
                    continue
                dataframe = pd.read_excel(path, skiprows=1, header=0)
                ax.plot(
                    dataframe.iloc[:, 0],
                    dataframe.iloc[:, column + 1],
                    color=cmap(norm(temperature)),
                    linewidth=1.2,
                )
                plotted = True
            ax.text(0.95, 0.06, ds_labels[edp][column], transform=ax.transAxes, fontsize=14, ha="right")
            ax.set_xlabel(r"$Sa(T_1)$", fontsize=18)
            if column == 0:
                ax.set_ylabel("超越概率", fontsize=18, fontname="SimSun")
            ax.set_xlim(0, 1.5)
            ax.set_xticks(np.arange(0, 1.51, 0.5))
            ax.set_ylim(0, 1)
            ax.tick_params(axis="both", direction="in", which="both", labelsize=14)

    if not plotted:
        plt.close(fig)
        raise FileNotFoundError(f"未找到 {model} 的易损性工作簿。")

    for row, marker in enumerate(("(a)", "(b)", "(c)")):
        axes[row, 0].text(-0.24, 0.5, marker, transform=axes[row, 0].transAxes, fontsize=15, va="center")

    colorbar_axes = fig.add_axes([0.88, 0.23, 0.018, 0.56])
    colorbar = fig.colorbar(scalar_map, cax=colorbar_axes)
    colorbar.set_label(f"温度 ({CELSIUS})", fontsize=18, fontname="SimSun", labelpad=12)
    colorbar.ax.tick_params(direction="in", labelsize=14)
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.08, top=0.97, wspace=0.26, hspace=0.28)
    output_path = OUT_DIR / f"易损性曲线面板_{model}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[生成] {output_path}")


def plot_fragility_surfaces(model: str) -> None:
    ds_labels = {
        "IDR": {1: "DS-1 (2.0%)", 2: "DS-2 (3.0%)", 3: "DS-3 (5.0%)"},
        "RIDR": {1: "DS-1 (0.2%)", 2: "DS-2 (0.5%)", 3: "DS-3 (1.0%)"},
        "PFA": {1: "DS-1 (0.5g)", 2: "DS-2 (1.0g)", 3: "DS-3 (1.5g)"},
    }
    x_common = np.linspace(0, 1.5, 301)
    temperature_dense = np.linspace(min(TEMPERATURES), max(TEMPERATURES), 121)

    for edp, labels in ds_labels.items():
        for ds_index, ds_label in labels.items():
            temperature_rows: list[float] = []
            probability_rows: list[np.ndarray] = []
            for temperature in TEMPERATURES:
                path = find_fragility_file(model, temperature, edp)
                if path is None:
                    continue
                dataframe = pd.read_excel(path, skiprows=1, header=0)
                x = dataframe.iloc[:, 0].astype(float).to_numpy()
                y = dataframe.iloc[:, ds_index].astype(float).to_numpy()
                order = np.argsort(x)
                temperature_rows.append(float(temperature))
                probability_rows.append(np.interp(x_common, x[order], y[order]))
            if not probability_rows:
                continue

            temperatures = np.asarray(temperature_rows)
            order = np.argsort(temperatures)
            temperatures = temperatures[order]
            z = np.asarray(probability_rows)[order]
            z_dense = np.vstack(
                [np.interp(temperature_dense, temperatures, z[:, index]) for index in range(z.shape[1])]
            ).T
            temperature_grid = np.tile(temperature_dense[:, None], (1, len(x_common)))
            sa_grid = np.tile(x_common[None, :], (len(temperature_dense), 1))

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_axes([0.01, 0.02, 0.97, 0.96], projection="3d")
            ax.plot_surface(
                sa_grid,
                temperature_grid,
                z_dense,
                cmap="plasma",
                norm=mcolors.Normalize(0, 1),
                rcount=z_dense.shape[0],
                ccount=z_dense.shape[1],
                linewidth=0,
                antialiased=False,
                shade=False,
            )
            ax.set_xlabel(r"$Sa(T_1)$", fontsize=25, labelpad=12)
            ax.set_ylabel(f"温度 ({CELSIUS})", fontsize=25, fontname="SimSun", labelpad=12)
            ax.set_zlabel("超越概率", fontsize=25, fontname="SimSun", labelpad=12)
            ax.set_xlim(0, 1.5)
            ax.set_ylim(min(TEMPERATURES), max(TEMPERATURES))
            ax.set_zlim(0, 1)
            ax.tick_params(axis="both", direction="in", which="both", labelsize=18)
            ax.view_init(elev=22, azim=-135)
            ax.text2D(0.82, 0.35, ds_label, transform=ax.transAxes, fontsize=20, ha="center")
            output_path = OUT_DIR / f"易损性曲面_{model}_{edp}_DS{ds_index}.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[生成] {output_path}")


def plot_regional_fragility_chinese(
    module: ModuleType,
    results: dict[str, dict[str, dict[str, object]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.8))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.98, bottom=0.12, wspace=0.18, hspace=0.24)
    for row, edp in enumerate(module.DAMAGE_STATES):
        axis_config = module.IM_AXIS_CONFIG[edp]
        for column, damage_state in enumerate(module.DAMAGE_STATES[edp]):
            ax = axes[row, column]
            item = results[edp][damage_state.name]
            for city_config in module.CITY_CONFIGS:
                city = city_config["city"]
                ax.plot(
                    item["im_values"],
                    item["regional_curves"][city],
                    color=city_config["color"],
                    linewidth=1.8,
                    label=city_config["label"],
                )
            ax.plot(
                item["im_values"],
                item["reference_curve"],
                color="#303030",
                linewidth=1.5,
                linestyle=(0, (4, 3)),
                label=celsius_label(int(module.REFERENCE_TEMP)),
            )
            ax.set_xlim(*axis_config["xlim"])
            ax.set_xticks(axis_config["xticks"])
            ax.set_ylim(0, 1.02)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.tick_params(axis="both", direction="in", which="both", labelsize=18)
            ax.set_xlabel(axis_config["xlabel"], fontsize=25)
            if column == 0:
                ax.set_ylabel("超越概率", fontsize=25, fontname="SimSun")
            ax.text(0.95, 0.90, damage_state.label, transform=ax.transAxes, ha="right", fontsize=18)
            ax.legend(loc="lower right", frameon=False, fontsize=18)
        axes[row, 0].text(-0.33, 0.5, f"({chr(97 + row)})", transform=axes[row, 0].transAxes, fontsize=20)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[生成] {output_path}")


def regenerate_regional_fragility(
    script_name: str,
    module_name: str,
    city_labels: dict[str, str],
    output_name: str,
) -> None:
    module = load_script_module(script_name, module_name)
    for config in module.CITY_CONFIGS:
        config["label"] = city_labels.get(config["city"], config["city"])
    _, region_weight_table = module.build_all_temperature_weights()
    results, _ = module.compute_regional_fragility_results(module.MODEL, region_weight_table)
    plot_regional_fragility_chinese(module, results, OUT_DIR / output_name)


def configure_regional_hazard(module: ModuleType, edp: str, output_dir: Path) -> None:
    module.EDP = edp
    config = module.EDP_CONFIGS[edp]
    module.ACTIVE_EDP_CONFIG = config
    module.PLOT_SCALE = float(config["plot_scale"])
    module.PLOT_XLABEL = str(config["plot_xlabel"])
    module.PLOT_UNIT = str(config["plot_unit"])
    module.OVERVIEW_FIGSIZE = tuple(config["overview_figsize"])
    module.ZOOM_FIGSIZE = tuple(config["zoom_figsize"])
    module.OVERVIEW_XLIM = tuple(config["overview_xlim"])
    module.OVERVIEW_YLIM = tuple(config["overview_ylim"])
    module.OVERVIEW_XTICKS = np.asarray(config["overview_xticks"], dtype=float)
    module.DS_CONFIGS = list(config["ds_configs"])
    module.ZOOM_AXIS_CONFIGS = dict(config["zoom_axis_configs"])
    module.ANNOTATION_OFFSETS = dict(config["annotation_offsets"])
    module.PAINT_DIR = output_dir
    module.OUTPUT_STEM = f"区域危险性_{module.MODEL}_{edp}"
    chinese_city_labels = {"Harbin": "哈尔滨", "Yinchuan": "银川", "Chongqing": "重庆", "20C": f"20 {CELSIUS}"}
    for city_config in module.CITY_CONFIGS:
        city_config["label"] = chinese_city_labels.get(city_config["city"], city_config["city"])
    module.CITY_LABELS = {str(config["city"]): str(config["label"]) for config in module.CITY_CONFIGS}


def regenerate_regional_hazard() -> None:
    module = load_script_module("plot_regional_hazard.py", "regional_hazard_chinese")
    output_dir = OUT_DIR / "区域危险性"
    output_dir.mkdir(parents=True, exist_ok=True)
    for edp in REGIONAL_HAZARD_EDPS:
        configure_regional_hazard(module, edp, output_dir)
        setup_chinese_style()
        module.main()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_chinese_style()

    if GENERATE_PUSHOVER:
        for model in MODELS:
            run_task(f"{model} Pushover 中文图", plot_pushover, model)
    if GENERATE_FRAGILITY_PANEL:
        for model in MODELS:
            run_task(f"{model} 易损性面板中文图", plot_fragility_panel, model)
    if GENERATE_FRAGILITY_SURFACE:
        for model in SURFACE_MODELS:
            run_task(f"{model} 易损性曲面中文图", plot_fragility_surfaces, model)
    if GENERATE_REGIONAL_FRAGILITY:
        run_task(
            "玉树-吐鲁番-文昌区域易损性中文图",
            regenerate_regional_fragility,
            "plot_regional_fragility_yushu_turpan_wenchang.py",
            "regional_fragility_yushu_chinese",
            {"Yushu": "玉树", "Turpan": "吐鲁番", "Wenchang": "文昌"},
            "区域易损性_玉树_吐鲁番_文昌.png",
        )
        run_task(
            "漠河-吐鲁番-三亚区域易损性中文图",
            regenerate_regional_fragility,
            "plot_regional_fragility_mohe_turpan_sanya.py",
            "regional_fragility_mohe_chinese",
            {"Mohe": "漠河", "Turpan": "吐鲁番", "Sanya": "三亚"},
            "区域易损性_漠河_吐鲁番_三亚.png",
        )
    if GENERATE_REGIONAL_HAZARD:
        run_task("区域危险性中文图", regenerate_regional_hazard)

    print(f"中文版图件输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
