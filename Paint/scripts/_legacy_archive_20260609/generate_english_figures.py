from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
PAINT_DIR = PROJECT_DIR / "Paint"
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = PAINT_DIR / "English"
TEMPERATURES = [-20, -10, 0, 10, 20, 30, 40]


def setup_style() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False


def copy_existing_figures() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffixes = {".png", ".jpg", ".jpeg", ".pdf"}
    for src in PAINT_DIR.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in suffixes:
            continue
        if OUT_DIR in src.parents:
            continue
        if "regional_temperature" in src.name.lower():
            continue
        rel = src.relative_to(PAINT_DIR)
        dst = OUT_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def read_pushover_data(folder_path: Path, h_building: float = 35600.0) -> tuple[np.ndarray, np.ndarray]:
    time = np.loadtxt(folder_path / "Time.out")
    weight = 0.0
    shear = np.zeros(len(time))
    for i in range(1, 1000):
        support_file = folder_path / f"Support{i}.out"
        if support_file.exists():
            data = np.loadtxt(support_file)
            weight += float(data[9, 1])
            shear += -data[:, 0]
    roof_disp = np.loadtxt(folder_path / "Disp9.out")
    return roof_disp * 100.0 / h_building, shear / weight


def plot_pushover(model: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for temp in TEMPERATURES:
        folder = PROJECT_DIR / "Output_data" / f"MC8_{model}_{temp}" / "MC8_PO" / "Pushover"
        if not folder.exists():
            continue
        x, y = read_pushover_data(folder)
        ax.plot(x, y, linewidth=2.2, label=fr"{temp}$^\circ$C")
    ax.set_xlabel("Roof drift (%)", fontsize=20, labelpad=10)
    ax.set_ylabel("Base shear coefficient (V/W)", fontsize=20, labelpad=10)
    ax.tick_params(direction="in", labelsize=18)
    ax.legend(loc="upper right", fontsize=15, frameon=True)
    ax.grid(linestyle="--", which="both", alpha=0.65)
    ax.set_xlim(0, 8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"Pushover_curve_{model}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_fragility_file(model: str, temp: int, edp: str) -> Path | None:
    folder = PROJECT_DIR / "Output_data" / f"MC8_{model}_{temp}" / "MC8_IDA_data_frag"
    if not folder.exists():
        return None
    candidates = [p for p in folder.glob(f"*_{edp}.xlsx") if "IDA" not in p.name and "姒傜巼" not in p.name]
    return candidates[0] if candidates else None


def plot_fragility_panel(model: str) -> None:
    dm_labels = {
        "IDR": ["DS-1 (2.0%)", "DS-2 (3.0%)", "DS-3 (5.0%)"],
        "RIDR": ["DS-1 (0.2%)", "DS-2 (0.5%)", "DS-3 (1.0%)"],
        "PFA": ["DS-1 (0.5g)", "DS-2 (1.0g)", "DS-3 (1.5g)"],
    }
    dms = ["IDR", "RIDR", "PFA"]
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=min(TEMPERATURES), vmax=max(TEMPERATURES))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(22, 16), sharey=True)
    plotted_any = False
    for i, dm in enumerate(dms):
        for j in range(3):
            ax = axes[i, j]
            for temp in TEMPERATURES:
                path = find_fragility_file(model, temp, dm)
                if path is None:
                    continue
                df = pd.read_excel(path, skiprows=1, header=0)
                ax.plot(df.iloc[:, 0], df.iloc[:, j + 1], color=cmap(norm(temp)), linewidth=1.1)
                plotted_any = True
            ax.text(0.95, 0.05, dm_labels[dm][j], transform=ax.transAxes, fontsize=30, ha="right", va="bottom")
            if j == 0:
                ax.set_ylabel("Exceedance probability", fontsize=30)
            ax.set_xlabel(r"$Sa(T_1)$", fontsize=30)
            ax.set_xlim(0, 1.5)
            ax.set_xticks(np.arange(0, 1.51, 0.5))
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=25, direction="in")

    for i, label in enumerate(["(a)", "(b)", "(c)"]):
        axes[i, 0].text(-0.3, 0.5, label, transform=axes[i, 0].transAxes, fontsize=28, va="center")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"Temperature ($^\circ$C)", fontsize=30)
    cbar.ax.tick_params(labelsize=30)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    if plotted_any:
        fig.savefig(OUT_DIR / f"Fragility_curves_{model}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_fragility_surfaces(model: str) -> None:
    ds_labels = {
        "IDR": {
            1: {"text": "DS-1  (2.0%)", "pos": (0.78, 0.35)},
            2: {"text": "DS-2  (3.0%)", "pos": (0.82, 0.35)},
            3: {"text": "DS-3  (5.0%)", "pos": (0.87, 0.35)},
        },
        "RIDR": {
            1: {"text": "DS-1  (0.2%)", "pos": (0.78, 0.35)},
            2: {"text": "DS-2  (0.5%)", "pos": (0.82, 0.35)},
            3: {"text": "DS-3  (1.0%)", "pos": (0.85, 0.35)},
        },
        "PFA": {
            1: {"text": "DS-1  (0.5g)", "pos": (0.78, 0.35)},
            2: {"text": "DS-2  (1.0g)", "pos": (0.80, 0.35)},
            3: {"text": "DS-3  (1.5g)", "pos": (0.80, 0.35)},
        },
    }
    x_common = np.linspace(0, 1.5, 301)
    temp_dense = np.linspace(min(TEMPERATURES), max(TEMPERATURES), 121)
    surf_x_ticks = np.arange(0, 1.5 + 0.1, 0.5)
    surf_y_ticks = np.arange(-20, 40 + 1, 10)
    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for dm, labels in ds_labels.items():
        for ds_index, label_config in labels.items():
            temp_rows: list[int] = []
            z_rows: list[np.ndarray] = []
            for temp in TEMPERATURES:
                path = find_fragility_file(model, temp, dm)
                if path is None:
                    continue
                df = pd.read_excel(path, skiprows=1, header=0)
                x = df.iloc[:, 0].astype(float).to_numpy()
                y = df.iloc[:, ds_index].astype(float).to_numpy()
                order = np.argsort(x)
                temp_rows.append(temp)
                z_rows.append(np.interp(x_common, x[order], y[order]))
            if not z_rows:
                continue

            temps = np.asarray(temp_rows, dtype=float)
            order = np.argsort(temps)
            temps = temps[order]
            z = np.asarray(z_rows)[order, :]
            z_dense = np.vstack([np.interp(temp_dense, temps, z[:, k]) for k in range(z.shape[1])]).T
            temp_grid = np.tile(temp_dense.reshape(-1, 1), (1, len(x_common)))
            sa_grid = np.tile(x_common.reshape(1, -1), (len(temp_dense), 1))

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_axes([0.01, 0.02, 0.97, 0.96], projection="3d")
            surf = ax.plot_surface(
                sa_grid,
                temp_grid,
                z_dense,
                cmap=cmap,
                norm=norm,
                rcount=z_dense.shape[0],
                ccount=z_dense.shape[1],
                linewidth=0,
                edgecolor="none",
                antialiased=False,
                shade=False,
            )
            ax.set_xlabel(r"$Sa(T_1)$", fontsize=25, labelpad=12)
            ax.set_ylabel(r"Temperature ($^\circ$C)", fontsize=25, labelpad=12)
            ax.set_zlabel("Exceedance probability (%)", fontsize=25, labelpad=12, rotation=-90)
            ax.set_xlim(0, 1.5)
            ax.set_xticks(surf_x_ticks)
            ax.set_xticklabels([""] + [f"{tick:.1f}" for tick in surf_x_ticks[1:]])
            ax.set_ylim(-20, 40)
            ax.set_yticks(surf_y_ticks)
            ax.set_zlim(0, 1)
            ax.tick_params(axis="x", direction="in", labelsize=18, pad=2)
            ax.tick_params(axis="y", direction="in", labelsize=18, pad=2)
            ax.tick_params(axis="z", direction="in", labelsize=18, pad=4)
            ax.view_init(elev=22, azim=-135)
            ax.text(0.0, -25, 0.0, f"{surf_x_ticks[0]:.1f}", fontsize=18, ha="center", va="top")
            ax.text2D(
                label_config["pos"][0],
                label_config["pos"][1],
                label_config["text"],
                transform=ax.transAxes,
                fontsize=23,
                fontweight=900,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=2.5),
            )
            fig.savefig(OUT_DIR / f"Fragility_Surface_{model}_{dm}_DS{ds_index}.png", dpi=300)
            plt.close(fig)


def run_module_function(script_name: str, function_name: str, *args) -> None:
    script = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(script.stem, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    getattr(module, function_name)(*args)


def regenerate_regional_fragility_old_regions() -> None:
    script = SCRIPT_DIR / "区域易损性.py"
    spec = importlib.util.spec_from_file_location("regional_fragility_old", script)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for city in module.CITY_CONFIGS:
        city["city_cn"] = city["city_en"]
    module.configure_matplotlib()
    plt.rcParams["font.family"] = "Times New Roman"
    bin_weight_table, region_weight_table = module.build_all_temperature_weights()
    results, curve_table = module.compute_regional_fragility_results(module.MODEL, region_weight_table)
    module.plot_regional_fragility_figure(results, OUT_DIR / f"Regional_fragility_curves_{module.MODEL}.png")


def regenerate_regional_hazard() -> None:
    script = SCRIPT_DIR / "区域危险性.py"
    text = script.read_text(encoding="utf-8")
    for edp in ["IDR", "RIDR", "PFA"]:
        patched = text.replace('EDP = "PFA"', f'EDP = "{edp}"')
        patched = patched.replace(
            'PAINT_DIR = PROJECT_ROOT / "Paint"',
            'PAINT_DIR = PROJECT_ROOT / "Paint" / "English"',
        )
        namespace = {"__file__": str(script), "__name__": "__main__"}
        exec(compile(patched, str(script), "exec"), namespace)


def main() -> None:
    setup_style()
    copy_existing_figures()
    for model in ["PFSDF", "SMABF"]:
        plot_pushover(model)
        plot_fragility_panel(model)
    plot_fragility_surfaces("PFSDF")
    regenerate_regional_fragility_old_regions()
    regenerate_regional_hazard()


if __name__ == "__main__":
    main()
