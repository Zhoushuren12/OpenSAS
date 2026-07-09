from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "Output_data"
PAINT_ROOT = PROJECT_ROOT / "Paint"

DEFAULT_MODELS = ("PFSDF", "SMABF")
EDP_TYPES = ("IDR", "RIDR", "PFA")
PARAMETERS = ("A", "B", "R2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总 PFSDF 和 SMABF 的 PSDM 拟合系数，并输出 Excel/PNG 表格。",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="需要汇总的模型名称，默认同时汇总 PFSDF 和 SMABF。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAINT_ROOT,
        help="输出目录，默认当前 Paint 文件夹。",
    )
    return parser.parse_args()


def extract_temperature(case_dir: Path, model: str) -> float:
    prefix = f"MC8_{model}_"
    if not case_dir.name.startswith(prefix):
        raise ValueError(f"鐩綍鍚嶄笉绗﹀悎棰勬湡: {case_dir.name}")
    temp_text = case_dir.name[len(prefix):]
    return float(temp_text)


def discover_case_dirs(model: str) -> list[tuple[float, Path]]:
    case_dirs: list[tuple[float, Path]] = []
    for case_dir in OUTPUT_ROOT.glob(f"MC8_{model}_*"):
        if not case_dir.is_dir():
            continue
        frag_dir = case_dir / "MC8_IDA_data_frag"
        if not frag_dir.exists():
            continue
        temp = extract_temperature(case_dir, model)
        case_dirs.append((temp, frag_dir))
    case_dirs.sort(key=lambda item: item[0])
    return case_dirs


def parse_probability_feature(file_path: Path) -> dict[str, float]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    result: dict[str, float] = {}
    for key in PARAMETERS:
        match = re.search(
            rf"^{re.escape(key)}\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            text,
            flags=re.MULTILINE,
        )
        if not match:
            raise ValueError(f"Parameter {key} not found in {file_path}")
        result[key] = float(match.group(1))
    return result


def format_temperature(temp: float) -> str:
    if temp.is_integer():
        return f"{int(temp)} \u00b0C"
    return f"{temp:g} \u00b0C"


def build_summary_dataframe(model: str) -> pd.DataFrame:
    case_dirs = discover_case_dirs(model)
    if not case_dirs:
        raise FileNotFoundError(f"未找到模型 {model} 的 MC8_IDA_data_frag 结果目录。")

    columns = pd.MultiIndex.from_product(
        [EDP_TYPES, ("A_t", "B_t", "R2")],
        names=["EDP", "Parameter"],
    )

    rows = []
    index = []
    for temp, frag_dir in case_dirs:
        row: list[float] = []
        for edp in EDP_TYPES:
            values = parse_probability_feature(frag_dir / f"概率特征_{edp}.out")
            row.extend([values["A"], values["B"], values["R2"]])
        rows.append(row)
        index.append(format_temperature(temp))

    df = pd.DataFrame(rows, index=index, columns=columns)
    df.index.name = "t"
    return df


def export_excel(summary_tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for model, df in summary_tables.items():
            df.to_excel(writer, sheet_name=model, merge_cells=True)
            sheet = writer.sheets[model]
            sheet.column_dimensions["A"].width = 12
            for col_idx in range(2, 11):
                sheet.column_dimensions[chr(64 + col_idx)].width = 12


def format_value(value: float, parameter: str) -> str:
    if parameter == "R2":
        return f"{value:.2f}"
    return f"{value:.3f}"


def draw_table_figure(df: pd.DataFrame, model: str, output_path: Path) -> None:
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    n_data_rows = len(df)
    row_h = 0.72
    title_h = 1.0
    header_h1 = 0.78
    header_h2 = 0.70
    bottom_pad = 0.35

    col_widths = [1.35] + [1.0] * 9
    x_edges = [0.0]
    for width in col_widths:
        x_edges.append(x_edges[-1] + width)
    total_w = x_edges[-1]
    total_h = title_h + header_h1 + header_h2 + n_data_rows * row_h + bottom_pad

    fig_w = 14
    fig_h = max(4.5, 0.42 * total_h + 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    title_y = total_h - 0.45
    ax.text(
        total_w / 2,
        title_y,
        f"{model} PSDM fitting coefficients (At, Bt)",
        ha="center",
        va="center",
        fontsize=18,
    )

    top_y = total_h - title_h
    mid_y = top_y - header_h1 - header_h2
    bottom_y = mid_y - n_data_rows * row_h

    ax.hlines(top_y, 0, total_w, colors="black", linewidth=2.0)
    ax.hlines(mid_y, 0, total_w, colors="black", linewidth=1.8)
    ax.hlines(bottom_y, 0, total_w, colors="black", linewidth=2.0)

    t_x = (x_edges[0] + x_edges[1]) / 2
    t_y = top_y - (header_h1 + header_h2) / 2
    ax.text(t_x, t_y, r"$t$", ha="center", va="center", fontsize=16)

    group_y = top_y - header_h1 / 2
    subheader_y = top_y - header_h1 - header_h2 / 2

    for group_idx, edp in enumerate(EDP_TYPES):
        start = 1 + group_idx * 3
        group_x = (x_edges[start] + x_edges[start + 3]) / 2
        ax.text(group_x, group_y, edp, ha="center", va="center", fontsize=17, fontweight="bold")

        for sub_idx, parameter in enumerate(("A_t", "B_t", "R2")):
            cell_x = (x_edges[start + sub_idx] + x_edges[start + sub_idx + 1]) / 2
            header_text = rf"${parameter}$" if parameter != "R2" else r"$R^2$"
            ax.text(cell_x, subheader_y, header_text, ha="center", va="center", fontsize=15)

    for row_idx, (row_label, row_series) in enumerate(df.iterrows()):
        y = mid_y - row_h * (row_idx + 0.5)
        ax.text(t_x, y, row_label, ha="center", va="center", fontsize=15)
        for group_idx, edp in enumerate(EDP_TYPES):
            for sub_idx, parameter in enumerate(("A_t", "B_t", "R2")):
                x = (x_edges[1 + group_idx * 3 + sub_idx] + x_edges[2 + group_idx * 3 + sub_idx]) / 2
                value = row_series[(edp, parameter)]
                ax.text(x, y, format_value(float(value), parameter), ha="center", va="center", fontsize=14)

    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_tables: dict[str, pd.DataFrame] = {}
    for model in args.models:
        df = build_summary_dataframe(model)
        summary_tables[model] = df
        draw_table_figure(df, model, output_dir / f"PSDM_fitting_coefficients_{model}.png")

    excel_path = output_dir / "PSDM_fitting_coefficients.xlsx"
    export_excel(summary_tables, excel_path)

    print(f"Generated Excel: {excel_path}")
    for model in summary_tables:
        print(f"Generated figure: {output_dir / f'PSDM_fitting_coefficients_{model}.png'}")


if __name__ == "__main__":
    main()
