import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

base_dir = Path('Output_data') 

# for i in range(2,10):  
             
# fig = plt.figure(figsize=(8, 6))
    # F = 2
    # file_path1 = base_dir / 'MC8_MRF' / 'MC8_TH_MCE_data' / f'{F}' / f'BeamSpring{i}_1R.out'
    # file_path2 = base_dir / 'MC8_SMRF' / 'MC8_TH_MCE_data' / f'{F}' / f'BeamSpring{i}_1R.out'
    # file_path3 = base_dir / 'MC8_SMAPFDF' / 'MC8_TH_MCE_data' / f'{F}' / f'BeamSpring{i}_1R.out'

    # data1 = np.loadtxt(file_path1)
    # M1 = data1[:, 0]
    # R1 = data1[:, 1]
    # # plt.plot(R1, M1, label='BeamSpring{}'.format(i))
    # plt.plot(R1, M1, label='MRF')

    # data2 = np.loadtxt(file_path2)
    # M2 = data2[:, 0]
    # R2 = data2[:, 1]
    # # plt.plot(R2, M2, label='BeamSpring{}'.format(i))
    # plt.plot(R2, M2, label='SMRF')

    # data3 = np.loadtxt(file_path3)
    # M3 = data3[:, 0]
    # R3 = data3[:, 1]
    # # plt.plot(R3, M3, label='BeamSpring{}'.format(i))
    # plt.plot(R3, M3, label='SMAPFDF')

    # data4 = np.loadtxt(file_path4)
    # M4 = data4[:, 0]
    # R4 = data4[:, 1]
    # plt.plot(R4, M4, label='BeamSpring')
             


# plt.plot(R1, F1, label='MRF')

# # plt.xlim(-0.4,0.4)
# plt.xlabel('Disp / mm', fontsize=14)
# plt.ylabel('Froce / N', fontsize=14)
# plt.tick_params(direction="in")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(linestyle='--', which='both')
# plt.title('Beam Hinge', fontsize=18)

# plt.tight_layout()
# plt.show()


def plot_beam_hinges(file_path, title):
    # 璇诲彇鏁版嵁锛堝惈鏍囬琛岋級
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    floors = data[:, 0]
    values = data[:, 1:]

    spans = 3  # 涓夎法
    plt.figure(figsize=(8, 6))

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o', 's']

    for i in range(spans):
        left = values[:, i*2]
        right = values[:, i*2 + 1]

        plt.plot(left, floors, label=f'Span {i+1} - Left', color=colors[i], linestyle='-', marker=markers[0])
        plt.plot(right, floors, label=f'Span {i+1} - Right', color=colors[i], linestyle='--', marker=markers[1])

    # 鍥捐〃璁剧疆
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(1, 9))
    plt.ylim(0.75, 8.25)
    # plt.xlim(0, 0.008)
    plt.xlabel('Plastic Rotation', fontsize=20)
    plt.ylabel('Floor', fontsize=20)
    plt.title(title, fontsize=20)
    plt.tick_params(labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()

def plot_beam_hinges_temp(model , level , title = None):

    temperature = ['-20','-10','0','10','20','30','40']

    for temp in temperature:

        data = base_dir / f'MC8_{model}_{temp}' / f'MC8_TH_{level}_data_out' / '结果统计' / f'梁铰_统计_50th.csv'

        floors = data[:, 0]
        values = data[:, 1:]

        plt.figure(figsize=(8, 6))
        plt.plot(floors, values, label=f'Temperature {temp}掳C')

    # 鍥捐〃璁剧疆
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(1, 9))
    plt.ylim(0.75, 8.25)
    # plt.xlim(0, 0.008)
    plt.xlabel('Plastic Rotation', fontsize=20)
    plt.ylabel('Floor', fontsize=20)
    if title:
        plt.title(title, fontsize=20)
    plt.tick_params(labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()


file_path1 = base_dir / 'MC8_MRF' / 'MC8_TH_MCE_data_out' / '结果统计' / '梁铰_统计_50th.csv'
file_path2 = base_dir / 'MC8_SMRF' / 'MC8_TH_MCE_data_out' / '结果统计' / '梁铰_统计_50th.csv'
file_path3 = base_dir / 'MC8_SMAPFDF' / 'MC8_TH_MCE_data_out' / '结果统计' / '梁铰_统计_50th.csv'

plot_beam_hinges(file_path1, 'MRF')
plot_beam_hinges(file_path2, 'SMRF')
plot_beam_hinges(file_path3, 'SMAPFDF')
plt.show()
