import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path

plt.rc('font',family='Times New Roman')

base_dir = Path('Output_data')

def plot_RIDR(file_path,  title, plot_mean=True, plot_median=False, plot_84th_percentile=False, limit=False):
    # 璇诲彇鏁版嵁
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    layers = data[1:, 0]  # 绗竴鍒椾负灞傛暟
    data_values = data[1:, 1:]  # 鍏朵綑鍒椾负鏁版嵁

    file_path = Path(file_path)
    file_path_2 = file_path.with_name(file_path.stem + '_统计.csv')

    mean = np.genfromtxt(file_path_2, delimiter=',', skip_header=1)[1:, 1]
    median = np.genfromtxt(file_path_2, delimiter=',', skip_header=1)[1:, 4]
    percentile_84 = np.genfromtxt(file_path_2, delimiter=',', skip_header=1)[1:, 5]

    # 缁樺浘
    plt.figure(figsize=(8, 6))
    for i in range(data_values.shape[1]):
        plt.plot(data_values[:, i], layers, color='gray', alpha=0.7, linewidth=1)  # 鐏拌壊鏇茬嚎

    # 缁樺埗鍧囧€肩嚎
    if plot_mean:
        plt.plot(mean, layers, color='red', linewidth=2, label='Mean')
        plt.scatter(mean, layers, color='red', s=50, zorder=5, marker='o')

    # 缁樺埗涓綅鏁扮嚎
    if plot_median:
        plt.plot(median, layers, color='blue', linewidth=2, label='Median')
        plt.scatter(median, layers, color='blue', s=50, zorder=5, marker='o')

    # 缁樺埗84鍒嗕綅鏁扮嚎
    if plot_84th_percentile:
        plt.plot(percentile_84, layers, color='green', linewidth=2, label='84th Percentile')
        plt.scatter(percentile_84, layers, color='green', s=50, zorder=5, marker='o')

    if limit:
        # 璁剧疆鍨傜洿铏氱嚎鍙婂叾鏍囩
        vertical_lines = [0.002, 0.005, 0.01]
        colors = ['black', 'black', 'black']
        sizes = [16, 16, 16]

        for i, (x_pos, color, size) in enumerate(zip(vertical_lines, colors, sizes)):
            plt.axvline(x=x_pos, color=color, linestyle='--', alpha=0.8)
            
            plt.text(x=x_pos+0.0002, y=1.2, s=f'DS{i+1}', 
                    color=color, 
                    fontsize=size,
                    ha='left',
                    va='top',)
        
    # 鍥捐〃璁剧疆
    plt.gca().invert_yaxis()
    plt.tick_params(axis='both', direction='in', which='both', labelsize=18)
    plt.ylim(0.75, 8.25)
    plt.yticks([1, 2, 3, 4, 5, 6, 7, 8])
    plt.xlim(0,0.02)
    plt.xlabel('RIDR', fontsize=25, labelpad=12)
    plt.ylabel('Floor', fontsize=25, labelpad=12)
    plt.title(title, fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.9])

    plt.show()

def RIDR_comparison(level: str, title, limit=True, plot_mean=False, plot_median=True, plot_84th_percentile=False,xlim=0.02):
    data_dict = {}
    layers = None

    folder_labels = {
        'MC8_MRF': 'MRF',
        'MC8_SMRF': 'SMRF',
        'MC8_SMAPFDF': 'SMAPFDF'
    }

    for folder_name, label in folder_labels.items():
        file_path = base_dir / folder_name / f'MC8_TH_{level}_data_out' / '结果统计' / '残余层间位移角.csv'
        if file_path.exists():
            try:
                data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
                if layers is None:
                    layers = data[1:, 0]
                values = data[1:, 1:]
                data_dict[label] = {
                    'median': np.nanmedian(values, axis=1),
                    'mean': np.nanmean(values, axis=1),
                    'p84': np.nanpercentile(values, 84, axis=1)
                }
            except Exception as e:
                print(f"[ERROR] Failed to read {label}: {e}")
        else:
            print(f"[SKIP] File not found: {file_path}")

    if not data_dict:
        print("[STOP] No valid data files found; skip plotting.")
        return

    plt.figure(figsize=(8, 6))

    if limit:
        vertical_lines = [0.002, 0.005, 0.01]
        for i, x in enumerate(vertical_lines):
            plt.axvline(x=x, color='black', linestyle='--', alpha=0.8)
            plt.text(x + 0.0002, y=1.2, s=f'DS{i+1}', fontsize=16, ha='left', va='top')

    for label, stats in data_dict.items():
        if plot_median:
            plt.plot(stats['median'], layers, label=label)
            plt.scatter(stats['median'], layers, marker='o')

        if plot_mean:
            plt.plot(stats['mean'], layers, linestyle='--', label=f'{label} Mean')
            plt.scatter(stats['mean'], layers, marker='x')

        if plot_84th_percentile:
            plt.plot(stats['p84'], layers, linestyle=':', label=f'{label} 84th')
            plt.scatter(stats['p84'], layers, marker='s')

    plt.gca().invert_yaxis()
    plt.tick_params(axis='both', direction='in', which='both', labelsize=18)
    plt.ylim(0.75, 8.25)
    plt.yticks(range(1, 9))
    plt.xlim(0, xlim)
    plt.xlabel('IDR', fontsize=20, labelpad=12)
    plt.ylabel('Floor', fontsize=20, labelpad=12)
    plt.legend(loc='upper right', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(title, fontsize=18)
    plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.9])
    plt.show()

def RIDR_comparison_temp(
    level: str, limit, title, 
    plot_mean=False, plot_median=True, plot_84th_percentile=False,
    xlim=0.1, model='SMAPFDF'  # 鏂板妯″瀷鍙傛暟
):
    data_dict = {}
    layers = None

    folder_labels = {
        'MC8_SMRF_-20': 'SMRF_-20',
        'MC8_SMRF_-10': 'SMRF_-0',
        'MC8_SMRF_0': 'SMRF_0',
        'MC8_SMRF_10': 'SMRF_10',
        'MC8_SMRF_20': 'SMRF_20',
        'MC8_SMRF_30': 'SMRF_30',
        'MC8_SMRF_40': 'SMRF_40',
        'MC8_SMAPFDF_-20': 'SMAPFDF_-20',
        'MC8_SMAPFDF_-10': 'SMAPFDF_-10',
        'MC8_SMAPFDF_0': 'SMAPFDF_0',
        'MC8_SMAPFDF_10': 'SMAPFDF_10',
        'MC8_SMAPFDF_20': 'SMAPFDF_20',
        'MC8_SMAPFDF_30': 'SMAPFDF_30',
        'MC8_SMAPFDF_40': 'SMAPFDF_40',
    }

    for folder, label in folder_labels.items():
        if not label.startswith(model):
            continue  # 鍙繚鐣欏綋鍓嶆ā鍨嬬殑鏇茬嚎

        file_path = base_dir / folder / f'MC8_TH_{level}_data_out' / '结果统计' / '残余层间位移角.csv'
        if file_path.exists():
            try:
                data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
                if layers is None:
                    layers = data[1:, 0]
                values = data[1:, 1:]
                data_dict[label] = {
                    'median': np.nanmedian(values, axis=1),
                    'mean': np.nanmean(values, axis=1),
                    'p84': np.nanpercentile(values, 84, axis=1)
                }
            except Exception as e:
                print(f"[ERROR] Failed to read {label}: {e}")
        else:
            print(f"[SKIP] File not found: {file_path}")

    if not data_dict:
        print("[STOP] No valid data files found; skip plotting.")
        return

    plt.figure(figsize=(8, 6))
    plt.axvline(x=limit, color='black', linestyle='--', alpha=0.5, label='Limit')

    for label, stats in data_dict.items():
        if plot_median:
            plt.plot(stats['median'], layers, linewidth=1, label=f'{label} Median')
            plt.scatter(stats['median'], layers, marker='o')

        if plot_mean:
            plt.plot(stats['mean'], layers, linewidth=1, label=f'{label} Mean')
            plt.scatter(stats['mean'], layers, marker='o')

        if plot_84th_percentile:
            plt.plot(stats['p84'], layers, linewidth=1, label=f'{label} 84th')
            plt.scatter(stats['p84'], layers, marker='o')

    plt.gca().invert_yaxis()
    plt.tick_params(axis='both', direction='in', which='both', labelsize=18)
    plt.ylim(0.75, 8.25)
    plt.yticks(range(1, 9))
    plt.xlim(0, xlim)
    plt.xlabel('RIDR', fontsize=20, labelpad=12)
    plt.ylabel('Floor', fontsize=20, labelpad=12)
    plt.legend(loc='upper right', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(title, fontsize=18)
    plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.9])
    plt.show()



Level = "CLE"
if Level == "DBE":
    file_path = base_dir / 'MC8_SMAPFDF' / 'MC8_TH_DBE_data_out' / '结果统计' / '残余层间位移角.csv'    
    RIDR_comparison(Level, 'DBE Level Story Drift Ratios')
    # plot_RIDR(file_path, 'DBE Story Drift Ratios', plot_mean=False, plot_median=True, plot_84th_percentile=False,limit=True)
elif Level == "MCE": 
    file_path = base_dir / 'MC8_SMRF' / 'MC8_TH_MCE_data_out' / '结果统计' / '残余层间位移角.csv'        
    RIDR_comparison(Level, 'MCE Level Story Drift Ratios')
    # plot_RIDR(file_path,  'SMRF Story Drift Ratios', plot_mean=False, plot_median=True, plot_84th_percentile=False,limit=True)
elif Level == "CLE":
    file_path = base_dir / 'MC8_SMAPFDF' / 'MC8_TH_CLE_data_out' / '结果统计' / '残余层间位移角.csv'        
    # plot_RIDR(file_path,  'SMAPFDF Story Drift Ratios', plot_mean=False, plot_median=True, plot_84th_percentile=False,limit=True)
    RIDR_comparison(Level, 'CLE Level Story Drift Ratios')
plt.show()
