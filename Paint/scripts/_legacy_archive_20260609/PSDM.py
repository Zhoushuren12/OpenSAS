# plot_psdm_with_params.py

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
plt.rc('font', family='Times New Roman')
plt.rcParams['axes.unicode_minus'] = False

# === Default data paths (relative to project root) ===

DATA_ROOT = Path("Output_data")




def read_psdm_excel(fp: str | Path):
    """
    Read the first four Excel columns as:
    0: scatter x of ln(DM)-ln(IM)
    1: scatter y of ln(DM)-ln(IM)
    2: fitted x
    3: fitted y

    If column names exist, try matching them; otherwise use the first four columns.
    Return (sx, sy, fx, fy) as numpy arrays.
    """
    df = pd.read_excel(fp)

    col_map_candidates = [
        (["ln(DM)-ln(IM)散点", "拟合值"], None),
    ]

    sx = sy = fx = fy = None

    if df.shape[1] >= 4:
        try:
            cols = [c.lower() for c in df.columns.astype(str)]
            def find_by_keywords(keywords):
                idxs = []
                for i, c in enumerate(cols):
                    if any(k in c for k in keywords):
                        idxs.append(i)
                return idxs

            scat_idx = find_by_keywords(["散点", "scatter"])
            fit_idx  = find_by_keywords(["拟合", "fitted", "fit"])

            if len(scat_idx) >= 2 and len(fit_idx) >= 2:
                sx = df.iloc[:, scat_idx[0]].to_numpy(dtype=float)
                sy = df.iloc[:, scat_idx[1]].to_numpy(dtype=float)
                fx = df.iloc[:, fit_idx[0]].to_numpy(dtype=float)
                fy = df.iloc[:, fit_idx[1]].to_numpy(dtype=float)
            else:
                sx = df.iloc[:, 0].to_numpy(dtype=float)
                sy = df.iloc[:, 1].to_numpy(dtype=float)
                fx = df.iloc[:, 2].to_numpy(dtype=float)
                fy = df.iloc[:, 3].to_numpy(dtype=float)
        except Exception:
            sx = df.iloc[:, 0].to_numpy(dtype=float)
            sy = df.iloc[:, 1].to_numpy(dtype=float)
            fx = df.iloc[:, 2].to_numpy(dtype=float)
            fy = df.iloc[:, 3].to_numpy(dtype=float)
    else:
        raise ValueError("Excel must contain at least 4 columns for scatter and fitted data.")

    mask_scatter = ~np.isnan(sx) & ~np.isnan(sy)
    mask_fit     = ~np.isnan(fx) & ~np.isnan(fy)

    return sx[mask_scatter], sy[mask_scatter], fx[mask_fit], fy[mask_fit]


def read_params_out(fp: str | Path):
    """
    Read parameters from a .out text file.
      A, B, R2, beta_D, beta_total, mean_p, std_p, median_p.
    """
    text = Path(fp).read_text(encoding="utf-8", errors="ignore")

    def find_float(key):
        m = re.search(rf"{re.escape(key)}\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", text)
        return float(m.group(1)) if m else None

    A          = find_float("A")
    B          = find_float("B")
    R2         = find_float("R2")
    beta_D     = find_float("beta_D")
    beta_total = find_float("beta_total")

    mean_p   = None
    std_p    = None
    median_p = None

    m = re.search(r"均值：\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", text)
    if m: mean_p = float(m.group(1))
    m = re.search(r"标准差：\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", text)
    if m: std_p = float(m.group(1))
    m = re.search(r"中位值：\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", text)
    if m: median_p = float(m.group(1))

    return dict(A=A, B=B, R2=R2, beta_D=beta_D, beta_total=beta_total,
                mean_p=mean_p, std_p=std_p, median_p=median_p)


def annotate_equation(ax, params: dict, loc="upper left", fontsize=12, DS="IDR"):
    """
    Add a text box showing the fitted equation and statistics.
    Arguments:
    - ax: matplotlib Axes 瀵硅薄
    - params: dictionary containing A, B, R2, beta_D, beta_total, mean_p, std_p, median_p.
    - loc: text box location, one of upper/lower left/right.
    """
    A = params.get("A")
    B = params.get("B")
    R2 = params.get("R2")
    beta_D = params.get("beta_D")
    beta_total = params.get("beta_total")
    mean_p = params.get("mean_p")
    std_p = params.get("std_p")
    median_p = params.get("median_p")

    lines = []

    # if R2 is not None:
    #     lines.append(r"$R^2 = {:.3f}$".format(R2))

    if A is not None and B is not None:
        lines.append(f"$\\ln({DS}) = {B:.3f}\\,\\ln(Sa) {A:+.3f}$")
        
    #     lines.append(r"$\beta_{{\mathrm{{D}}}} = {:.3f}$".format(beta_D))
    # if beta_total is not None:
    #     lines.append(r"$\beta_{{\mathrm{{total}}}} = {:.3f}$".format(beta_total))

    # extra = []
    # if mean_p is not None:
    #     extra.append("mean={:.3f}".format(mean_p))
    # if std_p is not None:
    #     extra.append("std={:.3f}".format(std_p))
    # if median_p is not None:
    #     extra.append("median={:.3f}".format(median_p))
    # if extra:
    #     lines.append("P(IM>0.1) stats: " + ", ".join(extra))

    text = "\n".join(lines)
    if not text:
        return

    loc_map = {
        "upper left":   dict(x=0.02, y=0.98, ha="left",  va="top"),
        "upper right":  dict(x=0.98, y=0.98, ha="right", va="top"),
        "lower left":   dict(x=0.02, y=0.02, ha="left",  va="bottom"),
        "lower right":  dict(x=0.98, y=0.02, ha="right", va="bottom"),
    }
    place = loc_map.get(loc, loc_map["upper left"])

    ax.text(
        place["x"], place["y"], text,
        transform=ax.transAxes,
        ha=place["ha"], va=place["va"],
        fontsize=fontsize,
        # bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8, lw=0.5)
    )




def main(model='SMAFBF', DS='RIDR', save_path=None,color = 'blue' ,show_plot=True):

    data_dir = DATA_ROOT / f"MC8_{model}" / "MC8_IDA_data_frag"

    excel_path = data_dir / f"概率需求模型_{DS}.xlsx"
    param_path = data_dir / f"概率特征_{DS}.out"
    

    sx, sy, fx, fy = read_psdm_excel(excel_path)

    params = read_params_out(param_path)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(sx, sy, 'o', markersize=6, color='gray', markeredgewidth=1)

    order = np.argsort(fx)
    ax.plot(np.array(fx)[order], np.array(fy)[order], '-', linewidth=2, label='Fitted relation', color=color)

    plt.tick_params(axis='both', direction='in', which='both', labelsize=25)
    ax.set_xlabel(f"ln(Sa)",fontsize=30, labelpad=12)
    ax.set_ylabel(f"ln({DS})",fontsize=30, labelpad=12)
    # ax.set_xlim(-2.5, 1)
    ax.set_ylim()
    # ax.set_yticks(np.arange(-6, -1, 1))
    ax.text(0.05, 0.95, f"{model}", transform=ax.transAxes,
            fontsize=28, va="top", ha="left")


    annotate_equation(ax, params, loc="lower right", fontsize=28, DS=DS)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    temp = [-20,-10,0,10,20,30,40]
    DSS = ['IDR', 'RIDR', 'PFA']
    for t in temp:
        for DS in DSS:
            model = f'SMABF_{t}'
            DS = DS
            save_path = f'Paint\\PSDM杈撳嚭\\PSDM_{model}_{DS}.png'

            main(model=model, DS=DS, save_path=save_path,color='RED', show_plot=False)
