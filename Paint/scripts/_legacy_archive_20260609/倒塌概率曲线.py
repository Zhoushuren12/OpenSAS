import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('font', family='Times New Roman')
plt.rc('mathtext', fontset='stix')

base_dir = Path('Output_data')


def get_y(x: list, y: list, x0: float, error: bool = True) -> float:
    if x0 < min(x):
        if error:
            raise ValueError(f'[Error] x0 < min(x) ({x0} < {min(x)})')
        return None
    if x0 > max(x):
        if error:
            raise ValueError(f'[Error] x0 > max(x) ({x0} > {max(x)})')
        return None
    for i in range(len(x) - 1):
        if x[i] == x0:
            return y[i]
        if x[i] < x0 <= x[i + 1]:
            k = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            return k * (x0 - x[i]) + y[i]
    raise ValueError('[Error] Intersection not found')


def _get_sa_mce_for_temp(root: Path, model: str, temp: int | str):
    data_MCE = np.loadtxt(root / 'Spectrum' / 'MCE Level Spectrum.txt')
    t_path = root / 'Output_data' / f'MC8_{model}_{temp}' / 'MC8_PO_out' / '鍛ㄦ湡(s).out'
    if not t_path.exists():
        # fallback to base model if temp folder is missing
        t_path = root / 'Output_data' / f'MC8_{model}' / 'MC8_PO_out' / '鍛ㄦ湡(s).out'
    T1 = float(np.loadtxt(t_path)[0])
    Sa_MCE = get_y(data_MCE[:, 0], data_MCE[:, 1], T1)
    return T1, Sa_MCE


def plot_exceedance_curves(file_paths: dict):
    colors = [
        "#2486cc",
        "#d62728",
        "#ff7f0e",
        "#2ca02c",
        "#cad627",
        "#d627d6",
        "#0eece1",
    ]

    plt.figure(figsize=(10, 6))
    cmr_text_lines = []
    target_sheet = 'P(IDR>0.1)'
    data_mce = np.loadtxt(Path('Spectrum') / 'MCE Level Spectrum.txt')
    root = Path(__file__).resolve().parents[2]

    for i, (model_name, path) in enumerate(file_paths.items()):
        xls = pd.ExcelFile(path)
        sheet_name = next(
            (s for s in xls.sheet_names if s.replace('&gt;', '>') == target_sheet),
            None
        )
        if sheet_name is None:
            raise ValueError(f"Worksheet named '{target_sheet}' not found in {path}")
        df = pd.read_excel(xls, skiprows=1, header=None, sheet_name=sheet_name)

        model_dir = path.parents[1]
        temp = path.parent.name.split('_')[-1]
        _, Sa_MCE = _get_sa_mce_for_temp(root, model_name, temp)

        x_line = df.iloc[:, 2] / Sa_MCE
        y_line = df.iloc[:, 3]

        color = colors[i % len(colors)]
        plt.plot(x_line, y_line, linestyle='-', color=color, label=f'{model_name} - Fit')

        cmr_json_path = path.parent / 'CMR.json'
        if cmr_json_path.exists():
            try:
                with open(cmr_json_path, 'r', encoding='utf-8') as f:
                    cmr_data = json.load(f)
                cmr_value = cmr_data.get('CMR', None)
                if cmr_value is not None:
                    cmr_text_lines.append(f'{model_name}_CMR = {cmr_value:.4f}')
            except Exception as e:
                print(f'鈿狅笍 鏃犳硶璇诲彇 {cmr_json_path}: {e}')
        else:
            print(f'File not found: {cmr_json_path}')

    if cmr_text_lines:
        cmr_text = '\n'.join(cmr_text_lines)
        plt.text(0.97, 0.01, cmr_text, fontsize=15, color='black',
                 transform=plt.gca().transAxes, ha='right', va='bottom')

    plt.xlabel('Sa(T1) / Sa_MCE(T1)')
    plt.ylabel('Probability of Exceedance')
    plt.title('Exceedance Curve Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_paths = {
        # 'PFSDF_-20': base_dir / 'MC8_PFSDF_-20' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        # 'PFSDF_-10': base_dir / 'MC8_PFSDF_-10' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        # 'PFSDF_0': base_dir / 'MC8_PFSDF_0' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        # 'PFSDF_10': base_dir / 'MC8_PFSDF_10' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        # 'PFSDF_20': base_dir / 'MC8_PFSDF_20' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        # 'PFSDF_30': base_dir / 'MC8_PFSDF_30' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        # 'PFSDF_40': base_dir / 'MC8_PFSDF_40' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',

        'SMABF_-20': base_dir / 'MC8_SMABF_-20' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        'SMABF_-10': base_dir / 'MC8_SMABF_-10' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        'SMABF_0': base_dir / 'MC8_SMABF_0' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        'SMABF_10': base_dir / 'MC8_SMABF_10' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        'SMABF_20': base_dir / 'MC8_SMABF_20' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        'SMABF_30': base_dir / 'MC8_SMABF_30' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
        'SMABF_40': base_dir / 'MC8_SMABF_40' / 'MC8_IDA_data_frag' / '鏄撴崯鎬ф洸绾縚IDR.xlsx',
    }

    plot_exceedance_curves(file_paths)
    # plot_exceedance_temp(
    #     model_select='PFSDF',
    #     DM='IDR',
    #     DS=('DS-1', 'DS-2', 'DS-3', 'DS-4'),
    #     temperatures=[-20, 0, 20, 40],
    #     plot_3d=False
    # )
