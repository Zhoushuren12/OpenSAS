from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('mathtext', fontset='stix')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "Output_data"
TEMPERATURE_DATA_DIR = PROJECT_ROOT / "daily_temperature_1974_2024"
PAINT_DIR = PROJECT_ROOT / "Paint"

MODEL = "PFSDF"
REFERENCE_TEMP = 20.0
TEMPERATURE_BIN_WIDTH = 2.5
ANALYZED_TEMPERATURES = np.array([-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0], dtype=float)

CITY_CONFIGS = [
    {"city_en": "Harbin", "city_cn": "哈尔滨", "color": "#5aa0de"},
    {"city_en": "Yinchuan", "city_cn": "银川", "color": "#9dcc65"},
    {"city_en": "Chongqing", "city_cn": "重庆", "color": "#f0ad4e"},
]

FIGURE_CAPTION = "Figure 4-11 Regional fragility curves: (a) IDR; (b) RIDR; (c) PFA"

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

DS_LABEL_POSITIONS = {
    ("IDR", "DS-3"): {"x": 0.06, "y": 0.94, "ha": "left", "va": "top"},
    ("RIDR", "DS-2"): {"x": 0.06, "y": 0.94, "ha": "left", "va": "top"},
}


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 1.0


def validate_reference_temperature() -> None:
    if REFERENCE_TEMP not in ANALYZED_TEMPERATURES:
        raise ValueError(
            f"Reference temperature {REFERENCE_TEMP:g} C is not in {ANALYZED_TEMPERATURES.tolist()}."
        )

def load_daily_temperatures(city_en: str) -> np.ndarray:
    csv_path = TEMPERATURE_DATA_DIR / f"{city_en}_daily_tavg_1974_2024.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Temperature data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"date", "tavg_C", "city"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing_columns)}")

    temperatures = df["tavg_C"].dropna().to_numpy(dtype=float)
    if temperatures.size == 0:
        raise ValueError(f"{csv_path.name} has no valid temperature data.")
    return temperatures


def build_temperature_bin_edges(temperatures: np.ndarray, bin_width: float) -> np.ndarray:
    temp_min = bin_width * math.floor(float(np.min(temperatures)) / bin_width)
    temp_max = bin_width * math.ceil(float(np.max(temperatures)) / bin_width)
    return np.arange(temp_min, temp_max + bin_width, bin_width, dtype=float)


def map_to_nearest_analyzed_temperature(bin_centers: np.ndarray) -> np.ndarray:
    distance = np.abs(bin_centers[:, None] - ANALYZED_TEMPERATURES[None, :])
    mapped_index = np.argmin(distance, axis=1)
    return ANALYZED_TEMPERATURES[mapped_index]


def check_probability_sum(probabilities: np.ndarray, label: str) -> None:
    prob_sum = float(np.sum(probabilities))
    if not np.isclose(prob_sum, 1.0, atol=1e-10):
        raise ValueError(f"{label} probability sum is not 1: {prob_sum:.12f}")


def build_city_temperature_weights(city_en: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    temperatures = load_daily_temperatures(city_en)
    bin_edges = build_temperature_bin_edges(temperatures, TEMPERATURE_BIN_WIDTH)
    counts, _ = np.histogram(temperatures, bins=bin_edges)

    total_count = int(np.sum(counts))
    if total_count == 0:
        raise ValueError(f"{city_en} has no valid data in temperature bins.")

    bin_probabilities = counts / total_count
    check_probability_sum(bin_probabilities, f"{city_en} bin probabilities")

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mapped_temperatures = map_to_nearest_analyzed_temperature(bin_centers)

    bin_df = pd.DataFrame(
        {
            "city": city_en,
            "bin_left_C": bin_edges[:-1],
            "bin_right_C": bin_edges[1:],
            "bin_center_C": bin_centers,
            "count": counts,
            "probability": bin_probabilities,
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
    aggregated_df.insert(0, "city", city_en)

    check_probability_sum(aggregated_df["region_probability"].to_numpy(float), f"{city_en} 鍖哄煙鏉冮噸")
    return bin_df, aggregated_df


def find_case_directory(model: str, temperature: float) -> Path:
    if float(temperature).is_integer():
        temp_text = str(int(temperature))
    else:
        temp_text = f"{temperature:g}"

    case_dir = OUTPUT_ROOT / f"MC8_{model}_{temp_text}"
    if not case_dir.exists():
        raise FileNotFoundError(f"Temperature case directory not found: {case_dir}")
    return case_dir


def parse_float_parameter(text: str, key: str) -> float:
    match = re.search(
        rf"^{re.escape(key)}\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Parameter not found: {key}")
    return float(match.group(1))


def _legacy_read_psdm_parameters_unused(model: str, temperature: float, edp: str) -> dict[str, float]:
    frag_dir = find_case_directory(model, temperature) / "MC8_IDA_data_frag"
    out_path = next((p for p in frag_dir.glob("*.out") if p.stem.endswith(f"_{edp}")), None)
    if out_path is None:
        raise FileNotFoundError(f"PSDM parameter file not found for {model} {temperature:g} C {edp}.")

    text = out_path.read_text(encoding="utf-8", errors="ignore")
    a_ln = parse_float_parameter(text, "A")
    b = parse_float_parameter(text, "B")
    beta_d_im = parse_float_parameter(text, "beta_D")
    return {
        "temp": float(temperature),
        "A": a_ln,
        "a": math.exp(a_ln),
        "b": b,
        "beta_D_IM": beta_d_im,
    }


def read_psdm_parameters(model: str, temperature: float, edp: str) -> dict[str, float]:
    frag_dir = find_case_directory(model, temperature) / "MC8_IDA_data_frag"
    candidate_paths = sorted(frag_dir.glob(f"*_{edp}.out"))
    if not candidate_paths:
        raise FileNotFoundError(f"PSDM parameter file not found for {model} {temperature:g} C {edp}.")

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

    checked_files = ", ".join(path.name for path in candidate_paths)
    raise ValueError(
        f"No A/B/beta_D parameters found for {model} {temperature:g} C {edp}: {checked_files}"
    )


def build_psdm_parameter_table(model: str) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for edp in DAMAGE_STATES:
        for temperature in ANALYZED_TEMPERATURES:
            params = read_psdm_parameters(model, temperature, edp)
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
        probability = float(row.region_probability)
        mapped_temperature = float(row.mapped_temperature_C)
        region_curve += probability * conditional_curves[mapped_temperature]
    return region_curve


def build_all_temperature_weights() -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_tables: list[pd.DataFrame] = []
    aggregated_tables: list[pd.DataFrame] = []

    for city in CITY_CONFIGS:
        bin_df, aggregated_df = build_city_temperature_weights(city["city_en"])
        bin_tables.append(bin_df)
        aggregated_tables.append(aggregated_df)

    return pd.concat(bin_tables, ignore_index=True), pd.concat(aggregated_tables, ignore_index=True)


def compute_regional_fragility_results(
    model: str,
    city_weight_table: pd.DataFrame,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]], pd.DataFrame]:
    results: dict[str, dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]] = {}
    curve_tables: list[pd.DataFrame] = []

    for edp, damage_states in DAMAGE_STATES.items():
        im_values = IM_AXIS_CONFIG[edp]["im_values"]
        edp_results: dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]] = {}

        params_by_temp = {
            float(temp): read_psdm_parameters(model, float(temp), edp) for temp in ANALYZED_TEMPERATURES
        }

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

            regional_curves: dict[str, np.ndarray] = {}
            for city in CITY_CONFIGS:
                city_weight = city_weight_table.loc[
                    city_weight_table["city"] == city["city_en"],
                    ["mapped_temperature_C", "region_probability"],
                ].copy()
                regional_curves[city["city_en"]] = build_regional_curve(city_weight, conditional_curves)

            reference_curve = conditional_curves[REFERENCE_TEMP]

            curve_df = pd.DataFrame({"EDP": edp, "DS": damage_state.name, "IM": im_values})
            for temperature in ANALYZED_TEMPERATURES:
                temp_label = f"temp_{int(temperature)}C" if float(temperature).is_integer() else f"temp_{temperature:g}C"
                curve_df[temp_label] = conditional_curves[float(temperature)]
            for city in CITY_CONFIGS:
                curve_df[f"{city['city_en']}_region"] = regional_curves[city["city_en"]]
            curve_df[f"reference_{int(REFERENCE_TEMP)}C"] = reference_curve
            curve_tables.append(curve_df)

            edp_results[damage_state.name] = {
                "im_values": im_values,
                "conditional_curves": conditional_curves,
                "regional_curves": regional_curves,
                "reference_curve": reference_curve,
                "damage_state": damage_state,
            }

        results[edp] = edp_results

    return results, pd.concat(curve_tables, ignore_index=True)


def plot_regional_fragility_figure(
    results: dict[str, dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]],
    output_path: Path,
) -> None:
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

            for city in CITY_CONFIGS:
                ax.plot(
                    im_values,
                    regional_curves[city["city_en"]],
                    color=city["color"],
                    linewidth=1.8,
                    label=city["city_cn"],
                )

            ax.plot(
                im_values,
                reference_curve,
                color="#303030",
                linewidth=1.5,
                linestyle=(0, (4, 3)),
                label=f"{int(REFERENCE_TEMP)}C",
            )

            ax.set_xlim(*axis_config["xlim"])
            ax.set_xticks(axis_config["xticks"])
            ax.set_ylim(0.0, 1.02)
            ax.set_yticks(np.arange(0.0, 1.01, 0.2))
            ax.tick_params(direction="in", top=True, right=True, labelsize=18)
            ax.set_xlabel(axis_config["xlabel"], fontsize=25, labelpad=8)

            if col_idx == 0:
                ax.set_ylabel("Exceedance probability", fontsize=18)

            if col_idx == 0:
                ax.set_ylabel("Exceedance probability", fontsize=25, labelpad=10)

            label_position = DS_LABEL_POSITIONS.get(
                (edp, damage_state.name),
                {"x": 0.95, "y": 0.90, "ha": "right", "va": "top"},
            )

            ax.text(
                label_position["x"],
                label_position["y"],
                damage_state.label,
                transform=ax.transAxes,
                ha=label_position["ha"],
                va=label_position["va"],
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

    # fig.text(0.5, 0.04, FIGURE_CAPTION, ha="center", fontsize=18)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def export_tables(
    model: str,
    bin_weight_table: pd.DataFrame,
    region_weight_table: pd.DataFrame,
    psdm_table: pd.DataFrame,
    curve_table: pd.DataFrame,
) -> None:
    PAINT_DIR.mkdir(parents=True, exist_ok=True)

    bin_weight_table.to_csv(
        PAINT_DIR / f"regional_temperature_bin_mapping_{model}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    region_weight_table.to_csv(
        PAINT_DIR / f"regional_temperature_weights_{model}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    psdm_table.to_csv(
        PAINT_DIR / f"regional_psdm_params_{model}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curve_table.to_csv(
        PAINT_DIR / f"regional_fragility_curves_{model}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    configure_matplotlib()
    validate_reference_temperature()

    bin_weight_table, region_weight_table = build_all_temperature_weights()
    psdm_table = build_psdm_parameter_table(MODEL)
    results, curve_table = compute_regional_fragility_results(MODEL, region_weight_table)

    export_tables(MODEL, bin_weight_table, region_weight_table, psdm_table, curve_table)
    plot_regional_fragility_figure(results, PAINT_DIR / f"Regional_fragility_curves_{MODEL}.png")

    print(f"Generated figure: {PAINT_DIR / f'Regional_fragility_curves_{MODEL}.png'}")
    print(f"Generated temperature-bin mapping: {PAINT_DIR / f'regional_temperature_bin_mapping_{MODEL}.csv'}")
    print(f"Generated regional temperature weights: {PAINT_DIR / f'regional_temperature_weights_{MODEL}.csv'}")
    print(f"Generated PSDM parameter table: {PAINT_DIR / f'regional_psdm_params_{MODEL}.csv'}")
    print(f"Generated regional fragility curve data: {PAINT_DIR / f'regional_fragility_curves_{MODEL}.csv'}")


if __name__ == "__main__":
    main()
