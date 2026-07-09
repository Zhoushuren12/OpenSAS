import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
plt.rc('font',family='Times New Roman')

base_dir = Path('Output_data') 


def plot_story_access(file_path, title, plot_mean=True, plot_median=False, plot_84th_percentile=False,skip_cols=None):
    # 璇诲彇鏁版嵁
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    if skip_cols is not None:
        data = np.delete(data, skip_cols, axis=1)

    layers = data[:, 0]  # 绗竴鍒椾负灞傛暟
    data_values = data[:, 1:]  # 鍏朵綑鍒椾负鏁版嵁


    mean = np.nanmean(data_values, axis=1)
    median = np.nanmedian(data_values, axis=1)
    percentile_84 = np.nanpercentile(data_values, 84, axis=1)


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


    # 鍥捐〃璁剧疆
    plt.gca().invert_yaxis()
    plt.tick_params(axis='both', direction='in', which='both', labelsize=18)
    plt.ylim(0.75, 8.25)
    plt.yticks([1, 2, 3, 4, 5, 6, 7, 8])
    plt.xlim()
    plt.xlabel('PFA(g)', fontsize=20, labelpad=12)
    plt.ylabel('Floor', fontsize=20, labelpad=12)
    plt.title(title, fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.9])

    plt.show()

def story_access_comparison(level: str,  title, plot_mean=False, plot_median=True, plot_84th_percentile=False):
    data_dict = {}
    layers = None

    folder_labels = {
        'MC8_MRF': 'MRF',
        'MC8_SMRF': 'SMRF',
        'MC8_SMAPFDF': 'SMAPFDF',
    }

    for folder, label in folder_labels.items():
        file_path = base_dir / folder / f'MC8_TH_{level}_data_out' / '结果统计' / '层加速度(g).csv'
        if file_path.exists():
            try:
                data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
                if layers is None:
                    layers = data[0:, 0]
                values = data[0:, 1:]
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
    plt.xlim(0,2)
    plt.xlabel('PFA(g)', fontsize=20, labelpad=12)
    plt.ylabel('Floor', fontsize=20, labelpad=12)
    plt.legend(loc='upper right', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(title, fontsize=18)
    plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.9])
    plt.show()

Level = "MCE"
if Level == "DBE":
    file_path = base_dir / 'MC8_SMAPFDF' / 'MC8_TH_DBE_data_out' / '结果统计' / '层加速度(g).csv'
    plot_story_access(file_path,  'DBE Level Story Drift Ratios', plot_mean=False, plot_median=True, plot_84th_percentile=False)
elif Level == "MCE":
    file_path = base_dir / 'MC8_SMAPFDF' / 'MC8_TH_MCE_data_out' / '结果统计' / '层加速度(g).csv'
    story_access_comparison(Level, 'MCE Level Story Drift Ratios', plot_mean=False, plot_median=True, plot_84th_percentile=False)
    # plot_story_access(file_path,  'SMRF Story Drift Ratios', plot_mean=False, plot_median=True, plot_84th_percentile=False,skip_cols=None)
elif Level == "CLE":
    file_path = base_dir / 'MC8_SMAPFDF' / 'MC8_TH_CLE_data_out' / '结果统计' / '层加速度(g).csv'
    plot_story_access(file_path, 'SMRF Story Drift Ratios', plot_mean=False, plot_median=False, plot_84th_percentile=False,skip_cols=None)

plt.show()
