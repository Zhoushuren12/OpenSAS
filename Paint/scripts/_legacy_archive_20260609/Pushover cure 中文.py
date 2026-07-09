from matplotlib.pylab import f
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
plt.rc('font', family='SimSun')
plt.rc('mathtext', fontset='stix')


base_dir = Path('Output_data' )

def read_pushover_data(folder_path, HBuilding=35600):
    folder = Path(folder_path)

    time_file = folder / 'Time.out'
    time = np.loadtxt(time_file)
    weight = 0
    shear = np.zeros(len(time))
    for i in range(1, 1000):
        if (folder / f'Support{i}.out').exists():
            data = np.loadtxt(folder / f'Support{i}.out')
            weight += data[9, 1]
            shear += -data[:, 0]
    shear_norm = shear / weight
    y = shear_norm
    # print(f'Weight: {weight:.2f} N')
    # Roof_disp = np.loadtxt(folder / f'SDR_Roof.out')
    Roof_disp = np.loadtxt(folder / f'Disp9.out')
    x = Roof_disp * 100 / HBuilding
    return x, y

fig=plt.figure(figsize=(8,6))
for temp in [-20,-10, 0,10, 20,30, 40]:
    folder_path = base_dir / f'MC8_SMABF_{temp}' / 'MC8_PO' / 'Pushover'
    x, y = read_pushover_data(folder_path)
    
    plt.plot(x, y, label=f'{temp}℃')

plt.xlabel('灞嬮《浣嶇Щ瑙?(%)', fontsize=20)
plt.ylabel('鍓噸姣?(V/W)', fontsize=20)
plt.tick_params(direction="in", labelsize=18)
plt.legend(loc='upper right', fontsize=16)
plt.grid(linestyle='--', which='both')
plt.ylim(0, 0.6)
plt.xlim(0, 8)
plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.9])
plt.savefig(f'Paint\\Pushover_curve_SMABF.png', dpi=300, bbox_inches='tight')
plt.show()
    

# folder_path1 =  base_dir / 'MC8_SMABF_-20' / 'MC8_PO' / 'Pushover'
# folder_path2 =  base_dir / 'MC8_SMABF_0' / 'MC8_PO' / 'Pushover'
# folder_path3 =  base_dir / 'MC8_SMABF_20' / 'MC8_PO' / 'Pushover'
# folder_path4 =  base_dir / 'MC8_SMABF_40' / 'MC8_PO' / 'Pushover'

# folder_path5 = base_dir / 'MC8_PFSDF_-20' / 'MC8_PO' / 'Pushover'
# folder_path6 = base_dir / 'MC8_PFSDF_0' / 'MC8_PO' / 'Pushover'
# folder_path7 = base_dir / 'MC8_PFSDF_20' / 'MC8_PO' / 'Pushover'
# folder_path8 = base_dir / 'MC8_PFSDF_40' / 'MC8_PO' / 'Pushover'

# # folder_path9 = base_dir / 'MC8_SMRF' / 'MC8_PO' / 'Pushover'


# x1, y1 = read_pushover_data(folder_path1)
# x2, y2 = read_pushover_data(folder_path2)
# x3, y3 = read_pushover_data(folder_path3)
# x4, y4 = read_pushover_data(folder_path4)

# x5, y5 = read_pushover_data(folder_path5)
# x6, y6 = read_pushover_data(folder_path6)
# x7, y7 = read_pushover_data(folder_path7)
# x8, y8 = read_pushover_data(folder_path8)

# # x9, y9 = read_pushover_data(folder_path9)

# fig=plt.figure(figsize=(8,6))
# ax=fig.add_subplot(111)


# ax.plot(x1,y1,label='SMABF_-20',linewidth=2)
# ax.plot(x2,y2,label='SMABF_0',linewidth=2)
# ax.plot(x3,y3,label='SMABF_20',linewidth=2)
# ax.plot(x4,y4,label='SMABF_40',linewidth=2)

# ax.plot(x5,y5,label='PFSDF_-20',linewidth=2,linestyle='-')
# ax.plot(x6,y6,label='PFSDF_0',linewidth=2,linestyle='-')
# ax.plot(x7,y7,label='PFSDF_20',linewidth=2,linestyle='-')
# ax.plot(x8,y8,label='PFSDF_40',linewidth=2,linestyle='-')

# # ax.plot(x9,y9,label='MRF',linewidth=2,linestyle=':')

# # ax.set_xlim(0,8)
# ax.set_ylim(0)
# # #ax.set_xticks(np.arange(0,0.11,0.02))
# # ax.set_yticks(np.arange(0,0.26,0.05))
# ax.set_ylabel('Vbase / Weigh',fontsize=14)
# ax.set_xlabel('Roof drift ( % )',fontsize=14)
# ax.tick_params(direction="in")
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.legend(loc='upper left',fontsize=12)
# ax.grid(linestyle='--',which='both')
# plt.show()


