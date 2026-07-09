from __future__ import annotations

"""漠河、吐鲁番和三亚区域易损性分析工具。

用途：将城市温度分布与温度相关易损性结合，生成区域易损性曲线和权重结果。
做法：读取温度数据与各温度工况 PSDM 参数，离散温度权重后计算城市综合易损性。
使用：修改“用户编辑区”中的城市数据、模型、温度分箱和输出目录后运行本文件。
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


from plot_common import (
    CELSIUS,
    celsius_label,
    configure_matplotlib as apply_common_style,
    fragility_file_name,
    normalize_temperature_label,
)

apply_common_style()
# =============================================================================
# 用户编辑区：项目路径、模型、温度分箱、城市数据和损伤状态
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "Output_data"
TEMPERATURE_DATA_DIR = Path(
    r"C:\Users\Admin\Desktop\考虑温度效应的PFSD\区域温度分析\Mohe_Turpan_Sanya\temperature_data\daily_temperature_1974_2024"
)
RESULT_DIR = PROJECT_ROOT / "Paint" / "Regional_Mohe_Turpan_Sanya_PFSDF"

MODEL = "PFSDF"
REFERENCE_TEMP = 20.0
TEMPERATURE_BIN_WIDTH = 2.5
ANALYZED_TEMPERATURES = np.array([-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0], dtype=float)

CITY_CONFIGS = [
    {"city": "Mohe", "label": "Mohe", "color": "#2f8de4"},
    {"city": "Turpan", "label": "Turpan", "color": "#f06a3a"},
    {"city": "Sanya", "label": "Sanya", "color": "#38a169"},
]

IM_AXIS_CONFIG = {
    "IDR": {
        "xlim": (0.0, 1.5),
        "xticks": np.arange(0.0, 1.51, 0.5),
        "xlabel": r"$Sa(T_1)$",
        "im_values": np.linspace(0.001, 1.5, 600),
    },
    "RIDR": {
        "xlim": (0.0, 1.2),
        "xticks": np.arange(0.0, 1.21, 0.4),
        "xlabel": r"$Sa(T_1)$",
        "im_values": np.linspace(0.001, 1.2, 600),
    },
    "PFA": {
        "xlim": (0.0, 1.5),
        "xticks": np.arange(0.0, 1.51, 0.5),
        "xlabel": r"$Sa(T_1)$",
        "im_values": np.linspace(0.001, 1.5, 600),
    },
}


@dataclass(frozen=True)
class DamageState:
    name: str
    dm: float
    beta_c: float
    label: str


DAMAGE_STATES = {
    "IDR": [
        DamageState("DS-1", 0.02, 0.3, "DS-1 (2.0%)"),
        DamageState("DS-2", 0.03, 0.3, "DS-2 (3.0%)"),
        DamageState("DS-3", 0.05, 0.3, "DS-3 (5.0%)"),
    ],
    "RIDR": [
        DamageState("DS-1", 0.002, 0.3, "DS-1 (0.2%)"),
        DamageState("DS-2", 0.005, 0.3, "DS-2 (0.5%)"),
        DamageState("DS-3", 0.01, 0.3, "DS-3 (1.0%)"),
    ],
    "PFA": [
        DamageState("DS-1", 0.5, 0.3, "DS-1 (0.5g)"),
        DamageState("DS-2", 1.0, 0.3, "DS-2 (1.0g)"),
        DamageState("DS-3", 1.5, 0.3, "DS-3 (1.5g)"),
    ],
}


# =============================================================================
# 用户编辑区结束
# =============================================================================


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 1.0


def parse_float_parameter(text: str, key: str) -> float:
    match = re.search(
        rf"^{re.escape(key)}\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Parameter {key} was not found.")
    return float(match.group(1))


def find_case_directory(model: str, temperature: float) -> Path:
    temp_text = str(int(temperature)) if float(temperature).is_integer() else f"{temperature:g}"
    case_dir = OUTPUT_ROOT / f"MC8_{model}_{temp_text}"
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing temperature case directory: {case_dir}")
    return case_dir


def read_psdm_parameters(model: str, temperature: float, edp: str) -> dict[str, float]:
    frag_dir = find_case_directory(model, temperature) / "MC8_IDA_data_frag"
    candidate_paths = sorted(frag_dir.glob(f"*_{edp}.out"))
    if not candidate_paths:
        raise FileNotFoundError(f"Missing PSDM parameter file for {model} {temperature:g} C {edp}.")

    for out_path in candidate_paths:
        text = out_path.read_text(encoding="utf-8", errors="ignore")
        try:
            a_ln = parse_float_parameter(text, "A")
            b = parse_float_parameter(text, "B")
            beta_d_im = parse_float_parameter(text, "beta_D")
        except ValueError:
            continue
        return {
            "temp": float(temperature),
            "A": a_ln,
            "a": math.exp(a_ln),
            "b": b,
            "beta_D_IM": beta_d_im,
        }

    raise ValueError(f"No readable A/B/beta_D values found in {frag_dir}.")


def load_daily_temperatures(city: str) -> np.ndarray:
    csv_path = TEMPERATURE_DATA_DIR / f"{city}_daily_tavg_1974_2024.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing temperature CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"date", "tavg_C", "city"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing_columns)}")

    temperatures = df["tavg_C"].dropna().to_numpy(dtype=float)
    if temperatures.size == 0:
        raise ValueError(f"{csv_path.name} has no valid temperatures.")
    return temperatures


def build_temperature_bin_edges(temperatures: np.ndarray) -> np.ndarray:
    temp_min = TEMPERATURE_BIN_WIDTH * math.floor(float(np.min(temperatures)) / TEMPERATURE_BIN_WIDTH)
    temp_max = TEMPERATURE_BIN_WIDTH * math.ceil(float(np.max(temperatures)) / TEMPERATURE_BIN_WIDTH)
    return np.arange(temp_min, temp_max + TEMPERATURE_BIN_WIDTH, TEMPERATURE_BIN_WIDTH, dtype=float)


def map_to_nearest_analyzed_temperature(bin_centers: np.ndarray) -> np.ndarray:
    distances = np.abs(bin_centers[:, None] - ANALYZED_TEMPERATURES[None, :])
    return ANALYZED_TEMPERATURES[np.argmin(distances, axis=1)]


def build_city_temperature_weights(city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    temperatures = load_daily_temperatures(city)
    bin_edges = build_temperature_bin_edges(temperatures)
    counts, _ = np.histogram(temperatures, bins=bin_edges)
    probabilities = counts / counts.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mapped_temperatures = map_to_nearest_analyzed_temperature(bin_centers)

    bin_df = pd.DataFrame(
        {
            "city": city,
            "bin_left_C": bin_edges[:-1],
            "bin_right_C": bin_edges[1:],
            "bin_center_C": bin_centers,
            "count": counts,
            "probability": probabilities,
            "mapped_temperature_C": mapped_temperatures,
        }
    )

    aggregated_df = (
        bin_df.groupby("mapped_temperature_C", as_index=False)["probability"].sum()
        .rename(columns={"probability": "region_probability"})
        .sort_values("mapped_temperature_C")
    )
    aggregated_df = pd.DataFrame({"mapped_temperature_C": ANALYZED_TEMPERATURES}).merge(
        aggregated_df,
        on="mapped_temperature_C",
        how="left",
    )
    aggregated_df["region_probability"] = aggregated_df["region_probability"].fillna(0.0)
    aggregated_df.insert(0, "city", city)

    prob_sum = float(aggregated_df["region_probability"].sum())
    if not np.isclose(prob_sum, 1.0, atol=1e-10):
        raise ValueError(f"{city} temperature weights do not sum to 1.0: {prob_sum:.12f}")

    return bin_df, aggregated_df


def build_all_temperature_weights() -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_tables = []
    aggregated_tables = []
    for city_config in CITY_CONFIGS:
        bin_df, aggregated_df = build_city_temperature_weights(city_config["city"])
        bin_tables.append(bin_df)
        aggregated_tables.append(aggregated_df)
    return pd.concat(bin_tables, ignore_index=True), pd.concat(aggregated_tables, ignore_index=True)


def conditional_fragility_curve(
    im_values: np.ndarray,
    a: float,
    b: float,
    beta_d_im: float,
    damage_state: DamageState,
) -> np.ndarray:
    total_beta = math.sqrt(beta_d_im**2 + damage_state.beta_c**2)
    ln_edp = np.log(a * np.power(im_values, b))
    z = (ln_edp - math.log(damage_state.dm)) / total_beta
    return norm.cdf(z)


def build_regional_curve(
    city_weights: pd.DataFrame,
    conditional_curves: dict[float, np.ndarray],
) -> np.ndarray:
    region_curve = np.zeros_like(next(iter(conditional_curves.values())))
    for row in city_weights.itertuples(index=False):
        region_curve += float(row.region_probability) * conditional_curves[float(row.mapped_temperature_C)]
    return region_curve


def build_psdm_parameter_table(model: str) -> pd.DataFrame:
    records = []
    for edp in DAMAGE_STATES:
        for temperature in ANALYZED_TEMPERATURES:
            params = read_psdm_parameters(model, float(temperature), edp)
            records.append(
                {
                    "model": model,
                    "edp": edp,
                    "temp": params["temp"],
                    "A": params["A"],
                    "a": params["a"],
                    "b": params["b"],
                    "beta_D_IM": params["beta_D_IM"],
                }
            )
    return pd.DataFrame(records)


def compute_regional_fragility_results(
    model: str,
    city_weight_table: pd.DataFrame,
) -> tuple[dict[str, dict[str, dict[str, object]]], pd.DataFrame]:
    results: dict[str, dict[str, dict[str, object]]] = {}
    curve_tables = []

    for edp, damage_states in DAMAGE_STATES.items():
        im_values = IM_AXIS_CONFIG[edp]["im_values"]
        params_by_temp = {
            float(temp): read_psdm_parameters(model, float(temp), edp) for temp in ANALYZED_TEMPERATURES
        }
        edp_results: dict[str, dict[str, object]] = {}

        for damage_state in damage_states:
            conditional_curves = {
                temp: conditional_fragility_curve(
                    im_values,
                    params["a"],
                    params["b"],
                    params["beta_D_IM"],
                    damage_state,
                )
                for temp, params in params_by_temp.items()
            }

            regional_curves = {}
            for city_config in CITY_CONFIGS:
                city = city_config["city"]
                city_weights = city_weight_table.loc[
                    city_weight_table["city"] == city,
                    ["mapped_temperature_C", "region_probability"],
                ].copy()
                regional_curves[city] = build_regional_curve(city_weights, conditional_curves)

            reference_curve = conditional_curves[REFERENCE_TEMP]

            curve_df = pd.DataFrame({"EDP": edp, "DS": damage_state.name, "IM": im_values})
            for temperature in ANALYZED_TEMPERATURES:
                curve_df[f"temp_{int(temperature)}C"] = conditional_curves[float(temperature)]
            for city_config in CITY_CONFIGS:
                city = city_config["city"]
                curve_df[f"{city}_region"] = regional_curves[city]
            curve_df[f"reference_{int(REFERENCE_TEMP)}C"] = reference_curve
            curve_tables.append(curve_df)

            edp_results[damage_state.name] = {
                "im_values": im_values,
                "regional_curves": regional_curves,
                "reference_curve": reference_curve,
                "damage_state": damage_state,
            }
        results[edp] = edp_results

    return results, pd.concat(curve_tables, ignore_index=True)


def plot_regional_fragility_figure(results: dict[str, dict[str, dict[str, object]]], output_path: Path) -> None:
    row_tags = ["(a)", "(b)", "(c)"]
    edp_order = list(DAMAGE_STATES.keys())

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.8))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.98, bottom=0.12, wspace=0.18, hspace=0.24)

    for row_idx, edp in enumerate(edp_order):
        axis_config = IM_AXIS_CONFIG[edp]
        for col_idx, damage_state in enumerate(DAMAGE_STATES[edp]):
            ax = axes[row_idx, col_idx]
            item = results[edp][damage_state.name]
            im_values = item["im_values"]
            regional_curves = item["regional_curves"]
            reference_curve = item["reference_curve"]

            for city_config in CITY_CONFIGS:
                city = city_config["city"]
                ax.plot(
                    im_values,
                    regional_curves[city],
                    color=city_config["color"],
                    linewidth=1.8,
                    label=city_config["label"],
                )

            ax.plot(
                im_values,
                reference_curve,
                color="#303030",
                linewidth=1.5,
                linestyle=(0, (4, 3)),
                label=celsius_label(int(REFERENCE_TEMP)),
            )

            ax.set_xlim(*axis_config["xlim"])
            ax.set_xticks(axis_config["xticks"])
            ax.set_ylim(0.0, 1.02)
            ax.set_yticks(np.arange(0.0, 1.01, 0.2))
            ax.tick_params(direction="in", top=True, right=True, labelsize=18)
            ax.set_xlabel(axis_config["xlabel"], fontsize=25, labelpad=8)

            if col_idx == 0:
                ax.set_ylabel("Exceedance probability", fontsize=25, labelpad=10)

            ax.text(
                0.95,
                0.90,
                damage_state.label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=18,
            )
            ax.legend(loc="lower right", frameon=False, fontsize=18, handlelength=2.6)

        axes[row_idx, 0].text(
            -0.33,
            0.50,
            row_tags[row_idx],
            transform=axes[row_idx, 0].transAxes,
            fontsize=20,
            fontweight="bold",
            va="center",
        )

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def export_tables(
    bin_weight_table: pd.DataFrame,
    region_weight_table: pd.DataFrame,
    psdm_table: pd.DataFrame,
    curve_table: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    bin_weight_table.to_csv(RESULT_DIR / "regional_temperature_bin_mapping_PFSDF_new_regions.csv", index=False, encoding="utf-8-sig")
    region_weight_table.to_csv(RESULT_DIR / "regional_temperature_weights_PFSDF_new_regions.csv", index=False, encoding="utf-8-sig")
    psdm_table.to_csv(RESULT_DIR / "regional_psdm_params_PFSDF_new_regions.csv", index=False, encoding="utf-8-sig")
    curve_table.to_csv(RESULT_DIR / "regional_fragility_curves_PFSDF_new_regions.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    configure_matplotlib()
    if REFERENCE_TEMP not in ANALYZED_TEMPERATURES:
        raise ValueError("REFERENCE_TEMP must be included in ANALYZED_TEMPERATURES.")

    bin_weight_table, region_weight_table = build_all_temperature_weights()
    psdm_table = build_psdm_parameter_table(MODEL)
    results, curve_table = compute_regional_fragility_results(MODEL, region_weight_table)

    export_tables(bin_weight_table, region_weight_table, psdm_table, curve_table)
    figure_path = RESULT_DIR / "Regional_fragility_curves_PFSDF_Mohe_Turpan_Sanya.png"
    plot_regional_fragility_figure(results, figure_path)

    print(f"Generated figure: {figure_path}")
    print(f"Generated output folder: {RESULT_DIR}")


if __name__ == "__main__":
    main()
