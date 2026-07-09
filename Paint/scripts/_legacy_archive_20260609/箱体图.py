import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from pathlib import Path
from itertools import cycle
import seaborn as sns
from matplotlib.patches import Rectangle
base_dir = Path('..') / 'Output_data' 
plt.rc('font',family='Times New Roman')

def read_data(model: str, level: str, rs: str):

    if rs == 'IDR':
        file_path = base_dir / f'MC8_{model}' / f'MC8_TH_{level}_data_out' / '结果统计' / '层间位移角.csv'
    elif rs == 'RIDR':
        file_path = base_dir / f'MC8_{model}' / f'MC8_TH_{level}_data_out' / '结果统计' / '残余层间位移角.csv'
    elif rs == 'PFA':
        file_path = base_dir / f'MC8_{model}' / f'MC8_TH_{level}_data_out' / '结果统计' / '层加速度(g).csv'
    else:
            raise ValueError("Invalid rs value. Must be IDR, RIDR, or PFA.")

    # 璇诲彇鏂囦欢鐨勭涓€琛岋紝浠ヤ究璁＄畻鍒楁暟
    with open(file_path, 'r') as file:
        first_line = file.readline()
        num_columns = len(first_line.split(','))  # 璁＄畻鍒楁暟
    
    # 璇诲彇鏁版嵁锛岃烦杩囩涓€琛屽拰绗竴鍒楋紝鎸夊垪璇诲彇
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=range(1, num_columns))
    
    data = data[~np.any(data == 0, axis=1)] 
    data = data.T
    data = data.flatten()
    if rs == 'IDR' or rs == 'RIDR':
        data = data*100
    return np.array(data)


def draw_custom_boxplot(data, temperature_labels, box_labels=None, 
                        box_colors =["#EC6A6A", "#70AEEC"],
                        ylabel='IDR (%)', ylim=None, figsize=(8,6), save_path=None,
                        show_mean=True, show_median=True):


    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')

    num_groups = len(temperature_labels)
    total_boxes = len(data)
    if total_boxes % num_groups != 0:
        raise ValueError("data length must be a multiple of temperature_labels length.")
    
    boxes_per_group = total_boxes // num_groups
    group_width = 1.0
    box_width = group_width / boxes_per_group

   # ------------------ 璁＄畻绠变綋浣嶇疆 ------------------
    shrink = 0.8
    positions = []
    group_edges = []
    for g in range(num_groups):
        group_start = g * group_width
        group_end   = group_start + group_width
        group_edges.append((group_start, group_end))
        group_center = (group_start + group_end)/2

        for i in range(boxes_per_group):
            offset = (i - (boxes_per_group-1)/2) * box_width * shrink
            positions.append(group_center + offset)

    group_centers = [np.mean(positions[i*boxes_per_group:(i+1)*boxes_per_group])
                     for i in range(num_groups)]
    
    if ylim is None:
        ylim = (0, max(np.max(d) for d in data) * 1.1)
    ax.set_ylim(*ylim)

    # ------------------ 棰滆壊 ------------------
    # cmap = plt.get_cmap("Set2")
    cmap = plt.get_cmap("Set1")
    group_color_cycle = cycle(cmap.colors)
    group_colors = [next(group_color_cycle) for _ in range(num_groups)]

    # box_colors =["#EC6A6A", "#70AEEC", "#EEC66A", "#EC6AE4", "#E4EC6A", "#6AE4EC", "#C66AEC"]
    box_colors = box_colors
    default_colors = [box_colors[i % boxes_per_group] for i in range(total_boxes)]

    # ------------------ 鑳屾櫙缁樺埗 ------------------
    background_alpha = 0.05
    for i, (left, right) in enumerate(group_edges):
        ax.axvspan(left, right, facecolor=group_colors[i], alpha=background_alpha, zorder=0)

    # 缁勯棿绔栫嚎
    for i in range(1, num_groups):
        prev_right = positions[i*boxes_per_group - 1] + box_width/2
        curr_left = positions[i*boxes_per_group] - box_width/2
        mid = (prev_right + curr_left) / 2
        ax.axvline(x=mid, color='black', linewidth=1)

    # 鐢荤绾垮浘
    boxprops = dict(linewidth=1.5, color='black')
    medianprops = dict(color='black', linewidth=2) if show_median else dict(color='none')
    meanprops = dict(marker='s', markerfacecolor='white', markeredgecolor='black', markersize=7) if show_mean else None

    bp = ax.boxplot(data, positions=positions, showmeans=show_mean, patch_artist=True,
                    whis=1.5, boxprops=boxprops, medianprops=medianprops, meanprops=meanprops,widths=0.2)

    # 璁剧疆绠变綋棰滆壊
    for patch, color in zip(bp['boxes'], default_colors):
        patch.set_facecolor(color)

    for flier in bp['fliers']:
        flier.set(marker='x', color='black', markersize=8)

    ax.set_xticks(group_centers)
    labels = ax.set_xticklabels(temperature_labels, fontsize=25)
    for label in labels:
        label.set_y(-0.02)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.tick_params(axis='both', direction='in', which='both', labelsize=25)
    ax.set_xlim(group_edges[0][0], group_edges[-1][1])
    ax.grid(False)

    ymin, ymax = ax.get_ylim()
    margin = (ymax - ymin) * 0.075
    offset = box_width * 0.25
    for i, dataset in enumerate(data):
        xpos = positions[i]
        if show_mean:
            mean_val = np.mean(dataset)
            yval = np.clip(mean_val, ymin + margin, ymax - margin)
            ax.text(xpos + offset, yval, f'{mean_val:.2f}',
                    ha='left', va='center', fontsize=20, rotation=90)
        if show_median:
            median_val = np.median(dataset)
            yval = np.clip(median_val, ymin + margin, ymax - margin)
            ax.text(xpos - offset, yval, f'{median_val:.2f}',
                    ha='right', va='center', fontsize=20, rotation=90)

    # 灏嗙浣撴爣绛剧Щ鑷虫爣棰樺尯鍩熷苟灞曠ず棰滆壊瀵圭収
    if box_labels is not None:
        if len(box_labels) != total_boxes:
            raise ValueError("box_labels length must match total number of boxes.")
        label_color_pairs = []
        seen_labels = set()
        for label, color in zip(box_labels, default_colors):
            if label in seen_labels:
                continue
            label_color_pairs.append((label, color))
            seen_labels.add(label)
        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=1.2)
            for _, color in label_color_pairs
        ]
        legend_labels = [label for label, _ in label_color_pairs]
        if legend_handles:
            ax.legend(
                legend_handles,
                legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.18),
                ncol=min(len(legend_handles), boxes_per_group),
                frameon=False,
                fontsize=18
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()




if __name__ == "__main__":

    np.random.seed(42)
    # data = [
    #     np.random.normal(1.90, 0.22, 22),
    #     np.random.normal(1.89, 0.31, 22),
    #     np.random.normal(1.82, 0.15, 22),
    #     np.random.normal(2.20, 0.4, 22)
    # ]
    data = [
    np.random.normal(1.8, 0.2, 22), np.random.normal(1.9, 0.3, 22),
    np.random.normal(2.0, 0.25, 22), np.random.normal(2.1, 0.35, 22),
    np.random.normal(2.2, 0.2, 22), np.random.normal(2.4, 0.4, 22),
    np.random.normal(2.5, 0.2, 22), np.random.normal(2.6, 0.4, 22),
]
    positions = [1, 2, 3, 4 ]
    custom_colors=["#E45353", '#66B3FF', "#28D128", "#CC843B", '#FFB3E6', '#FF6666', '#66FFCC', '#FFCC66']
    temperature_labels = ['20掳C', '0掳C', '-20掳C', '-40掳C']
    box_labels = ['SMRF', 'SMAPFDF', 'SMRF', 'SMAPFDF','SMRF', 'SMAPFDF','SMRF', 'SMAPFDF']


    draw_custom_boxplot(data, temperature_labels, box_labels=box_labels, box_colors=custom_colors,
                        ylabel='IDR (%)',show_mean=False, show_median=True,)

