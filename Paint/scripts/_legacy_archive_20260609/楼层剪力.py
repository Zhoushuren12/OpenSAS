import re
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
plt.rc('font',family='Times New Roman')

base_dir = Path('Output_data' )

def plot_Force(file_path,  title, plot_mean=True, plot_median=False, plot_84th_percentile=False):
    # 璇诲彇鏁版嵁
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    layers = data[0:, 0]  # 绗竴鍒椾负灞傛暟


    file_path = Path(file_path)


    mean = np.genfromtxt(file_path, delimiter=',', skip_header=1)[0:, 1]
    median = np.genfromtxt(file_path, delimiter=',', skip_header=1)[0:, 4]
    percentile_84 = np.genfromtxt(file_path, delimiter=',', skip_header=1)[0:, 5]


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
    plt.tick_params(axis='both', direction='in', which='both', labelsize=18)
    plt.ylim(0.75, 8.25)
    plt.yticks([1, 2, 3, 4, 5, 6, 7, 8])
    # plt.xlim(0,0.02)
    plt.xlabel('RIDR', fontsize=25, labelpad=12)
    plt.ylabel('Floor', fontsize=25, labelpad=12)
    plt.title(title, fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.9])



def real_shear(model, SMAPFDF=False):
    shear = np.zeros(8)
    folder = base_dir / f'MC8_{model}' / 'MC8_PO' / 'Pushover'

    for story in range(1, 9):
        shear_story = None
        brace_story = None

        for file in folder.iterdir():
            if f'Shear{story}_' in file.name:
                data = np.loadtxt(file)[:, 0] / 1000
                if shear_story is not None:
                    shear_story += data
                else:
                    shear_story = data

            if SMAPFDF and f'SMA{story}_' in file.name:
                data = np.loadtxt(file)[:, 0] / 1000
                data = np.abs(data) 
                if brace_story is not None:
                    brace_story += data
                else:
                    brace_story = data

        # 澶勭悊缂哄け鏁版嵁锛堝鏋滄枃浠朵笉瀛樺湪鎴栨暟鎹负绌猴級
        if shear_story is None:
            shear_story = np.zeros(1)
        if SMAPFDF and brace_story is None:
            brace_story = np.zeros(1)

        if SMAPFDF:
            total_shear = max(np.abs(abs(shear_story) + brace_story))
            brace_max = max(np.abs(brace_story))
            print(f"Story {story} - Shear: {total_shear:.2f} kN, Brace: {brace_max:.2f} kN")
        else:
            total_shear = max(np.abs(shear_story))

        shear[story - 1] = total_shear

    return shear



# Level = "DBE"
# if Level == "DBE":
#     plot_Force(file_path, 'DBE Level Story Drift Ratios',plot_mean=True, plot_median=True, plot_84th_percentile=False)
    
# elif Level == "MCE":
#     plot_Force(file_path, 'MCE Level Story Drift Ratios',plot_mean=True, plot_median=True, plot_84th_percentile=False)
    
# elif Level == "CLE":
    
#     plot_Force(file_path, 'MCE Level Story Drift Ratios',plot_mean=True, plot_median=True, plot_84th_percentile=False)
# plt.show()
read_shear1 = real_shear('MRF')
read_shear2 = real_shear('MRF_R')
read_shear3 = real_shear('SMRF', SMAPFDF=True)
read_shear4 = real_shear('SMAPFDF', SMAPFDF=True)
print(read_shear1,'\n',
      read_shear2,'\n',
      read_shear3,'\n',
      read_shear4,'\n',         
      )
