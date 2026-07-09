import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import builtins
import pandas as pd

base_dir = Path('Output_data' )

def read_K_th(model,level='DBE'):
    if level == 'DBE':
        file_path1 = base_dir / model / 'MC8_TH_DBE_data_out' / '结果统计' / '楼层剪力_统计.csv' 
        file_path2 = base_dir / model / 'MC8_TH_DBE_data_out' / '结果统计' / '层间位移角_统计.csv'
    elif level == 'MCE':
        file_path1 = base_dir / model / 'MC8_TH_MCE_data_out' / '结果统计' / '楼层剪力_统计.csv' 
        file_path2 = base_dir / model / 'MC8_TH_MCE_data_out' / '结果统计' / '层间位移角_统计.csv' 
    elif level == 'CLE':
        file_path1 = base_dir / model / 'MC8_TH_CLE_data_out' / '结果统计' / '楼层剪力_统计.csv' 
        file_path2 = base_dir / model / 'MC8_TH_CLE_data_out' / '结果统计' / '层间位移角_统计.csv'
    else:
        raise ValueError("Invalid level. Choose from 'DBE', 'MCE', or 'CLE'.")

    data1 = np.genfromtxt(file_path1,delimiter=',', skip_header=1)
    data2 = np.genfromtxt(file_path2, delimiter=',', skip_header=1) 
    layers = data1[:, 0]  # 绗竴鍒椾负灞傛暟
    K_values = data1[:, 4] / data2[1:, 4]

    plt.figure(figsize=(10, 6))
    plt.plot(K_values, layers, marker='o', color='blue', label='K')
    plt.xlabel('Stiffness (K)', fontsize=14)
    plt.ylabel('Floor', fontsize=14)
    plt.title(f'Story Stiffness Ratios at {level} Level', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
   

def read_K_Po(model):
    story_height = [5400, 4200, 4200, 4200, 4200, 4200, 4200, 4200]  # 鍗曚綅锛歮m
    story_height = [h / 1000 for h in story_height]  # 杞崲涓虹背
    folder = base_dir / model / 'MC8_PO' / 'Pushover'
    stiffness = []      
    Force = []  
    disp = []
    for story in range(1, len(story_height) + 1):
        H = story_height[story - 1]
        shear_story = None
        for file in folder.iterdir():
            if f'Shear{story}_' in file.name:
                data = np.loadtxt(file)[:, 0] / 1000
                if shear_story is not None:
                    shear_story += data
                else:
                    shear_story = data
        V = max(abs(shear_story))
        data = pd.read_csv(folder / f'SDR{story}.out', header=None).to_numpy()[:, 0]
        data_max= max(abs(data))
        Disp = data_max * H
        K = V / Disp  # 鍗曚綅 kN/m
        stiffness.append(K)
        Force.append(V)
        disp.append(Disp)
    return stiffness , Force, disp
 



   
model = 'MC8_SMRF'  # 鏇挎崲鎴愪綘瀹為檯鐨勬ā鍨嬬洰褰曞悕
stiffness , Force, disp= read_K_Po(model)
plt.figure(figsize=(8, 6))
plt.plot(stiffness,range(1, 9), marker='o', color='blue', label='Stiffness')
plt.plot(Force, range(1, 9), marker='x', color='red', label='Force')
plt.plot(disp, range(1, 9), marker='s', color='green', label='Displacement')
plt.legend()
plt.show()
# read_K('MC8_MRF', level='DBE')
