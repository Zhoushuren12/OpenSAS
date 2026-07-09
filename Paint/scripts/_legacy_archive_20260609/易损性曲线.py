import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from pathlib import Path

plt.rc('font', family='Times New Roman')
plt.rc('mathtext', fontset='stix')


def get_y(x: list, y: list, x0: float, error: bool = True) -> float:
    """Get the y value at x = x0 by linear interpolation."""
    if x0 < min(x):
        if error:
            raise ValueError(f'[Error] x0 < min(x) ({x0} < {min(x)})')
        return None
    if x0 > max(x):
        if error:
            raise ValueError(f'[Error] x0 > max(x) ({x0} > {max(x)})')
        return None
    for i in range(len(x) - 1):
        if x[i] == x0:
            return y[i]
        if x[i] < x0 <= x[i + 1]:
            k = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            return k * (x0 - x[i]) + y[i]
    raise ValueError('[Error] Intersection not found')

def _get_sa_mce_for_temp(root: Path, model: str, temp: int | str):
    data_MCE = np.loadtxt(root / 'Spectrum' / 'MCE Level Spectrum.txt')
    t_path = root / 'Output_data' / f'MC8_{model}_{temp}' / 'MC8_PO_out' / '鍛ㄦ湡(s).out'
    if not t_path.exists():
        # fallback to base model if temp folder is missing
        t_path = root / 'Output_data' / f'MC8_{model}' / 'MC8_PO_out' / '鍛ㄦ湡(s).out'
    T1 = float(np.loadtxt(t_path)[0])
    Sa_MCE = get_y(data_MCE[:, 0], data_MCE[:, 1], T1)
    return T1, Sa_MCE

def frag_curve_temp():
    models = ['SMABF']
    DMs = ['IDR', 'RIDR', 'PFA']
    temperatures = [-20,-10, 0,10, 20,30, 40]
    cmap = plt.get_cmap('coolwarm')

    DM_labels = {
        'IDR': ['DS-1 (1.5%)', 'DS-2 (2.5%)', 'DS-3 (3.75%)'],
        'RIDR': ['DS-1 (0.5%)', 'DS-2 (1.0%)', 'DS-3 (2.0%)'],
        'PFA': ['DS-1 (0.2g)', 'DS-2 (0.5g)', 'DS-3 (1.0g)'],
    }

    root = Path(__file__).resolve().parents[2]

    for model in models:
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9), sharey=True)

        norm = mcolors.Normalize(vmin=min(temperatures), vmax=max(temperatures))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for i, DM in enumerate(DMs):
            for j in range(3):
                ax = axes[i, j]

                for temp in temperatures:
                    color = cmap(norm(temp))
                    _, Sa_MCE = _get_sa_mce_for_temp(root, model, temp)
                    path = root / 'Output_data' / f'MC8_{model}_{temp}' / \
                        'MC8_IDA_data_frag' / f'鏄撴崯鎬ф洸绾縚{DM}.xlsx'
                    if not path.exists():
                        continue

                    df = pd.read_excel(path, skiprows=1, header=0)
                    y = df.iloc[:, j + 1]
                    x = df.iloc[:, 0] / Sa_MCE
                    ax.plot(x, y, color=color, linewidth=1)

                ax.text(0.95, 0.05, DM_labels[DM][j], transform=ax.transAxes,
                        fontsize=15, ha='right', va='bottom')

                if j == 0:
                    ax.set_ylabel('Exceeding probability', fontsize=16)
                ax.set_xlabel(r'$Sa(T_1) / Sa_{MCE}(T_1)$', fontsize=16)

        labels = ['(a)', '(b)', '(c)']
        for i, label in enumerate(labels):
            axes[i, 0].text(-0.3, 0.5, label, transform=axes[i, 0].transAxes,
                           fontsize=18, fontweight='bold', va='center')

        for ax in axes.flat:
            ax.set_xlim(0, 3)
            ax.set_xticks(np.arange(0, 5.1, 1))
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=14, direction='in')

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Temperatures (掳C)', fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()

def frag_curve_3d():
    models = ['PFSDF']
    DMs = ['IDR', 'RIDR', 'PFA']
    temperatures = [-20, -10, 0, 10, 20, 30, 40]
    DM_labels = {
        'IDR': ['DS-1 (1.5%)', 'DS-2 (2.5%)', 'DS-3 (3.75%)'],
        'RIDR': ['DS-1 (0.5%)', 'DS-2 (1.0%)', 'DS-3 (2.0%)'],
        'PFA': ['DS-1 (0.2g)', 'DS-2 (0.5g)', 'DS-3 (1.0g)'],
    }
    cmap = cm.viridis

    root = Path(__file__).resolve().parents[2]

    frag_surfaces = [[[] for _ in range(3)] for _ in range(len(DMs))]
    Sa_values = None

    for dm_idx, DM in enumerate(DMs):
        for model in models:
            for temp in temperatures:
                path = root / 'Output_data' / f'MC8_{model}_{temp}' / \
                    'MC8_IDA_data_frag' / f'鏄撴崯鎬ф洸绾縚{DM}.xlsx'
                if not path.exists():
                    continue

                try:
                    df = pd.read_excel(path, skiprows=1, header=0)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue

                if DM in ['IDR', 'RIDR']:
                    DS_cols = [1,2,3]
                else:
                    DS_cols = [1, 2, 3]

                if df.shape[1] <= max(DS_cols):
                    print(f"Insufficient columns in {path}")
                    continue

                _, Sa_MCE = _get_sa_mce_for_temp(root, model, temp)
                x = df.iloc[:, 0] / Sa_MCE
                if Sa_values is None:
                    Sa_values = x.values

                for j, col in enumerate(DS_cols):
                    if col >= df.shape[1]:
                        continue
                    y = df.iloc[:, col].values
                    if len(y) != len(Sa_values):
                        continue
                    frag_surfaces[dm_idx][j].append((temp, y))

    fig = plt.figure(figsize=(14, 10), facecolor='honeydew')
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for dm_idx, DM in enumerate(DMs):
        for ds_idx in range(3):
            ax = fig.add_subplot(3, 3, dm_idx * 3 + ds_idx + 1, projection='3d')
            temp_vals = []
            z_vals = []

            for temp_value, y in frag_surfaces[dm_idx][ds_idx]:
                temp_vals.append(np.full_like(Sa_values, temp_value))
                z_vals.append(y)

            if not temp_vals:
                continue

            Temp_grid = np.array(temp_vals)
            Sa_grid = np.tile(Sa_values, (Temp_grid.shape[0], 1))
            Z_grid = np.array(z_vals)

            ax.plot_surface(
                Sa_grid, Temp_grid, Z_grid,
                facecolors=cmap(norm(Z_grid)), rstride=1, cstride=1,
                linewidth=0, antialiased=False, shade=False
            )

            ax.set_xlabel(r'$Sa(T_1)/Sa_{MCE}(T_1)$', fontsize=12, labelpad=5)
            ax.set_ylabel('Temperature (掳C)', fontsize=12, labelpad=5)
            ax.set_zlabel('Exceeding Probability', fontsize=12, labelpad=5)
            ax.set_zlim(0, 1)
            ax.set_xlim(0, 3.1)
            ax.set_xticks(np.arange(0, 3.1, 1))
            ax.set_ylim(-20, 40)
            ax.set_yticks(np.arange(-20, 41, 20))
            ax.text(
                np.max(Sa_grid), np.min(Temp_grid), 0,
                DM_labels[DM][ds_idx],
                fontsize=12, weight='bold', color='black',
                ha='right', va='bottom'
            )

            ax.view_init(elev=20, azim=-135)
            ax.xaxis._axinfo['grid']['color'] = (0.7, 0.7, 0.7, 0.3)
            ax.yaxis._axinfo['grid']['color'] = (0.7, 0.7, 0.7, 0.3)
            ax.zaxis._axinfo['grid']['color'] = (0.7, 0.7, 0.7, 0.3)
            ax.tick_params(axis='x', pad=2)
            ax.tick_params(axis='y', pad=2)
            ax.tick_params(labelsize=12, direction='in')

    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


if __name__ == '__main__':
    frag_curve_temp()
    # frag_curve_3d()
