import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

model = "PFSDF"

# 寤鸿锛氬湪寰幆寮€濮嬪墠鍒濆鍖栦竴涓敾甯冿紝鍙互鎸囧畾澶у皬
plt.figure(figsize=(10, 6)) 
# temperatures = [-20, 0, 20, 40]
temperatures = [-20,-10, 0,10, 20,30, 40]
for temp in temperatures:
                            
    path = Path('Output_data') / f'MC8_{model}_{temp}' / 'MC8_IDA_data_frag' / f'鏄撴崯鎬ф洸绾縚IDR.xlsx'
    
    data = pd.read_excel(path, skiprows=1, sheet_name=0)
    IM = data.iloc[:, 0]
    DM1 = data.iloc[:, 3]
    
    plt.plot(IM, DM1, label=f'{temp}掳C')


plt.xlabel('IM')       
plt.ylabel('Value')   
plt.grid(True)
plt.legend() 
plt.ylim(0,1)
plt.xlim(0,2.5)
plt.show()
