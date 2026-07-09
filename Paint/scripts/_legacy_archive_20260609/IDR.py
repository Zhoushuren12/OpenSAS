import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import importlib; draw_custom_boxplot = importlib.import_module("\u7bb1\u4f53\u56fe").draw_custom_boxplot
import pandas as pd
from pathlib import Path
from typing import Iterable, List
import numpy as np
base_dir = Path('Output_data' )

RS_FILE_MAP = {
    "IDR": "层间位移角.csv",
    "RIDR": "残余层间位移角.csv",
    "PFA": "层加速度(g).csv",
    "DCF": "DCF.csv",
}

YLABEL_MAP = {
    "IDR": "层间位移角.csv",
    "RIDR": "残余层间位移角.csv",
    "PFA": "层加速度(g).csv",
    "DCF": "DCF",
}

def read_data(model: str, level: str, rs: str, skip_cols=None) -> np.ndarray:

    if rs == 'IDR':
        file_path = base_dir / f'MC8_{model}' / f'MC8_TH_{level}_data_out' / '结果统计' / '层间位移角.csv'
    elif rs == 'RIDR':
        file_path = base_dir / f'MC8_{model}' / f'MC8_TH_{level}_data_out' / '结果统计' / '残余层间位移角.csv'
    elif rs == 'PFA':
        file_path = base_dir / f'MC8_{model}' / f'MC8_TH_{level}_data_out' / '结果统计' / '层加速度(g).csv'
    else:
        raise ValueError("Invalid rs value. Must be 'IDR', 'RIDR', or 'PFA'.")

    # 璇诲彇鏂囦欢鐨勭涓€琛岋紝浠ヤ究璁＄畻鍒楁暟
    with open(file_path, 'r') as file:
        first_line = file.readline()
        num_columns = len(first_line.split(','))  # 璁＄畻鍒楁暟
    
    # 璇诲彇鏁版嵁锛岃烦杩囩涓€琛屽拰绗竴鍒楋紝鎸夊垪璇诲彇
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=range(1, num_columns))

    if skip_cols is not None:
        skip_cols = np.array(skip_cols) - 1  
        data = np.delete(data, skip_cols, axis=1)

    data = data[~np.any(data == 0, axis=1)] 
    data = np.array(data.T)
    data = np.max(data, axis=1) 

    if rs == 'IDR' or rs == 'RIDR':
        data = data*100
    return np.array(data)

def collect_dataset(models: List[str], levels: List[str], rs: str, skip_cols: Iterable[int] | None = None) -> List[np.ndarray]:
    """
    Read data arrays for each model and intensity level combination.
    """
    dataset: List[np.ndarray] = []
    for level in levels:
        for model in models:
            dataset.append(read_data(model, level, rs, skip_cols))
    return dataset

if __name__ == "__main__":

   
    DS = 'IDR'
    Level = 'ERE'

    data1 = read_data('PFSDF_-20', Level,DS,skip_cols=None)
    data2 = read_data('PFSDF_-10', Level,DS,skip_cols=None)
    data3 = read_data('PFSDF_0', Level,DS,skip_cols=None)
    data4 = read_data('PFSDF_10', Level,DS,skip_cols=None)
    data5 = read_data('PFSDF_20', Level,DS,skip_cols=None)
    data6 = read_data('PFSDF_30', Level,DS,skip_cols=None)
    data7 = read_data('PFSDF_40', Level,DS,skip_cols=None)

    data8 = read_data('SMABF_-20', Level,DS,skip_cols=None)
    data9 = read_data('SMABF_-10', Level,DS,skip_cols=None)
    data10 = read_data('SMABF_0', Level,DS,skip_cols=None)
    data11 = read_data('SMABF_10', Level,DS,skip_cols=None)
    data12 = read_data('SMABF_20', Level,DS,skip_cols=None)
    data13 = read_data('SMABF_30', Level,DS,skip_cols=None)
    data14 = read_data('SMABF_40', Level,DS,skip_cols=None)

    # data = [data1,data5,data2,data6,data3,data7,data4,data8]
    # data = [data1,data2,data3,data4,data5,data6,data7,data8]

    # data = [data8,data9,data10,data11,data12,data13,data14]
    data = [data1,data8,data2,data9,data3,data10,data4,data11,data5,data12,data6,data13,data7,data14]

    temperature_labels = ['-20℃','-10℃','0℃','10℃','20℃','30℃','40℃']
    save_path = f'Output_data/{DS}.png'
    draw_custom_boxplot(data,  temperature_labels, 
                        ylabel='IDR (%)',ylim=(0,10),show_mean=False,show_median=True,
                        save_path=None)
