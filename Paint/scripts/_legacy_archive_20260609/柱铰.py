from math import floor
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


base_dir = Path('Output_data') 


# for i in range(1, 9):  
    # fig = plt.figure(figsize=(8, 6))
#     file_path = base_dir / 'MC8_SMRF_20' / 'MC8_PO' / 'Pushover' / f'ColSpring{i}_1T.out'
#     data = np.loadtxt(file_path)
#     M = data[:, 0]
#     R = data[:, 1]
#     plt.plot(R, M, label=f'ColSpring{i}')

# # F = 2
# # G = 9
# # file_path1 = base_dir / 'MC8_MRF' / 'MC8_TH_CLE_data' / f'{G}' / f'ColSpring{F}_1B.out'
#     file_path1 = base_dir / 'MC8_SMRF_20' / 'MC8_PO' / 'Pushover' / f'ColSpring{i}_1T.out'
# # file_path2 = base_dir / 'MC8_SMRF' / 'MC8_TH_CLE_data' / f'{G}' / f'ColSpring{F}_1B.out'
# # file_path3 = base_dir / 'MC8_SMAPFDF' / 'MC8_TH_CLE_data' / f'{G}' / f'ColSpring{F}_1B.out'
# # file_path4 = base_dir / 'MC8_SMRF' / 'MC8_PO' / 'Pushover' / f'ColSpring{F}_1B.out'

#     data1 = np.loadtxt(file_path1)
#     M1 = data1[:, 0]
#     R1 = data1[:, 1]
#     # plt.plot(R1, M1, label='BeamSpring{}'.format(i))
#     plt.plot(R1, M1, label='MRF')

# # data2 = np.loadtxt(file_path2)
# # M2 = data2[:, 0]
# # R2 = data2[:, 1]
# # # plt.plot(R2, M2, label='BeamSpring{}'.format(i))
# # plt.plot(R2, M2, label='SMRF')

# # data3 = np.loadtxt(file_path3)
# # M3 = data3[:, 0]
# # R3 = data3[:, 1]
# # # plt.plot(R3, M3, label='BeamSpring{}'.format(i))
# # plt.plot(R3, M3, label='SMAPFDF')

# # data4 = np.loadtxt(file_path4)
# # M4 = data4[:, 0]
# # R4 = data4[:, 1]
# # plt.plot(R4, M4, label='ColSpring')

# # 璁剧疆鍧愭爣杞磋寖鍥达紙鍙€夛級
# ax.set_xlim()
# # ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
# ax.set_xlabel('Disp / mm', fontsize=14)
# ax.set_ylabel('Froce / N', fontsize=14)

# ax.tick_params(direction="in")
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# # 鏀剧疆鍥句緥
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# # 娣诲姞缃戞牸
# ax.grid(linestyle='--', which='both')
# plt.title('Colmun Hinge', fontsize=16)
# plt.tight_layout()
# plt.show()

def plot_col_hinges(file_path, title):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=(1, 2, 3, 4))

    if data.shape[0] != 16:
        raise ValueError(f"期望16行数据（每层上下），但读取到 {data.shape[0]} 行，请检查 CSV 格式。")

    floors = np.arange(1, 9)
    col1, col2, col3, col4 = [], [], [], []
    y1, y2, y3, y4 = [], [], [], []

    for i in range(8):
        idx_down = i * 2
        idx_up = i * 2 + 1

        col1.append(data[idx_down][0]); y1.append(floors[i] - 0.1)
        col2.append(data[idx_down][1]); y2.append(floors[i] - 0.1)
        col3.append(data[idx_down][2]); y3.append(floors[i] - 0.1)
        col4.append(data[idx_down][3]); y4.append(floors[i] - 0.1)
        col1.append(data[idx_up][0]); y1.append(floors[i] + 0.1)
        col2.append(data[idx_up][1]); y2.append(floors[i] + 0.1)
        col3.append(data[idx_up][2]); y3.append(floors[i] + 0.1)
        col4.append(data[idx_up][3]); y4.append(floors[i] + 0.1)

    # 鐢诲浘
    plt.figure(figsize=(8, 6))
    plt.plot(col1, y1, 'o-', label='Column 1')
    plt.plot(col2, y2, 's-', label='Column 2')
    plt.plot(col3, y3, '^-', label='Column 3')
    plt.plot(col4, y4, 'd-', label='Column 4')

    # plt.gca().invert_yaxis()
    plt.yticks(np.arange(1, 9), fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel('Plastic Rotation', fontsize=18)
    plt.ylabel('Story', fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

def plot_col_hinges_temp(model , level , title = None):


    temperature = ['-20','-10','0','10','20','30','40']

    for temp in temperature:

        file_path = base_dir / f'MC8_{model}_{temp}' / f'MC8_TH_{level}_data_out' / '结果统计' / f'柱铰_统计_50th.csv'

        data = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=(1, 2, 3, 4))

        floors = np.arange(1, 9)
        max_rotations = []
        for i in range(8):
            idx_down = i * 2
            idx_up = i * 2 + 1

            layer_data = np.vstack([data[idx_down], data[idx_up]])  # shape: (2, 4)
            max_each_col = np.max(layer_data, axis=0)
            max_each_floor = np.max(max_each_col)
            max_rotations.append(max_each_floor)
            plt.plot(max_rotations, floors, 'o-', color='darkred', label='Max Plastic Rotation per Floor')

    # 鐢诲浘
    plt.figure(figsize=(8, 6))
    # plt.gca().invert_yaxis()
    plt.yticks(np.arange(1, 9), fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel('Plastic Rotation', fontsize=18)
    plt.ylabel('Story', fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()



file_path1 = base_dir / 'MC8_MRF' / 'MC8_TH_MCE_data_out' / '结果统计' / '柱铰_统计_50th.csv'
file_path2 = base_dir / 'MC8_SMRF' / 'MC8_TH_MCE_data_out' / '结果统计' / '柱铰_统计_50th.csv'
file_path3 = base_dir / 'MC8_SMAPFDF' / 'MC8_TH_MCE_data_out' / '结果统计' / '柱铰_统计_50th.csv'

plot_col_hinges(file_path1, 'MRF')
plot_col_hinges(file_path2, 'SMRF')
plot_col_hinges(file_path3, 'SMAPFDF')
plt.show()
