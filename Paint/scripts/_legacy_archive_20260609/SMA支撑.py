from pathlib import Path
from pickle import FALSE
import re
import numpy as np
import matplotlib.pyplot as plt

# --- 璁剧疆瀛椾綋 ---
plt.rc('font', family='Times New Roman')
plt.rc('mathtext', fontset='stix')

# ==============================================================================
#                               鏍稿績缁樺浘鍑芥暟
# ==============================================================================

def plot_hysteresis_curves_debug(
    file_paths,
    labels=None,
    *,
    x_col=1, y_col=0,
    x_scale=1.0, y_scale=1/1000,
    title=None,
    xlabel="Displacement (mm)", ylabel="Force (kN)",
    figsize=(8, 6), grid=True, lw=1.6,
    xlim=None, ylim=None,
    save_path=None, show=True,
    skip_bad_lines=False
):
    paths = [Path(p) for p in file_paths]
    if labels is None: labels = [p.stem for p in paths]
    
    plt.figure(figsize=figsize)
    plotted = 0

    for p, lab in zip(paths, labels):
        if not p.exists():
            print(f"[WARN] File not found, skipped: {p}")
            continue
        try:
            if skip_bad_lines:
                arr = np.genfromtxt(p, dtype=float, invalid_raise=False)
            else:
                arr = np.loadtxt(p, dtype=float)
            if arr.ndim == 1: arr = arr.reshape(1, -1)
            if arr.shape[1] <= max(x_col, y_col): continue

            x = arr[:, x_col] * x_scale
            y = arr[:, y_col] * y_scale
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            if len(x) > 0:
                plt.plot(x, y, label=lab, linewidth=lw)
                plotted += 1
        except Exception as e:
            print(f"[ERROR] 璇诲彇 {lab} 澶辫触: {e}")
            continue

    if title: plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid: plt.grid(True, alpha=0.3)
    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)
    if plotted > 0: plt.legend()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    if show: plt.show()
    else: plt.close()

def _pad_limits(lo, hi, pad=0.05):
    if not np.isfinite(lo) or not np.isfinite(hi): return (0.0, 1.0)
    span = hi - lo if hi != lo else (abs(lo) if lo!=0 else 1.0)
    return (lo - span * pad, hi + span * pad)

# ==============================================================================
#           銆愰噸鐐逛慨鏀广€戞敮鎸侀儴鍒嗙己澶辩殑 GIF 鐢熸垚鍑芥暟
# ==============================================================================
def make_hysteresis_gif_from_frames(
    frame_file_groups,
    save_path,
    *,
    frame_labels=None,
    series_labels=None,
    series_colors=None,
    x_col=1, y_col=0,
    x_scale=1.0, y_scale=1/1000,
    title=None,
    xlabel="Displacement (mm)", ylabel="Force (kN)",
    figsize=(8, 6), grid=True, lw=1.6,
    xlim=None, ylim=None,
    fps=2, dpi=120,
    skip_bad_lines=False, show=False
):
    save_path = Path(save_path)
    if not frame_file_groups: return

    n_series = len(frame_file_groups[0])
    if series_labels is None: series_labels = [f"Series {i+1}" for i in range(n_series)]
    if series_colors is None: series_colors = [None] * n_series
    if frame_labels is None: frame_labels = [str(i) for i in range(len(frame_file_groups))]

    frames_x, frames_y = [], []
    frames_valid_lab = []

    print(f"Reading {len(frame_file_groups)} frames...")
    
    g_xmin, g_xmax = float('inf'), float('-inf')
    g_ymin, g_ymax = float('inf'), float('-inf')
    
    has_any_valid_data = False

    for group, f_lab in zip(frame_file_groups, frame_labels):
        current_frame_xs = []
        current_frame_ys = []
        
        frame_has_data = False
        
        for p in group:
            p = Path(p)
            data_x, data_y = None, None # 榛樿涓虹┖

            # 灏濊瘯璇诲彇
            if p.exists():
                try:
                    if skip_bad_lines:
                        arr = np.genfromtxt(p, dtype=float, invalid_raise=False)
                    else:
                        arr = np.loadtxt(p, dtype=float)
                    
                    if arr.ndim == 1: arr = arr.reshape(1, -1)
                    
                    if arr.shape[1] > max(x_col, y_col):
                        x = arr[:, x_col] * x_scale
                        y = arr[:, y_col] * y_scale
                        mask = np.isfinite(x) & np.isfinite(y)
                        x, y = x[mask], y[mask]
                        
                        if len(x) > 0:
                            data_x, data_y = x, y
                            g_xmin, g_xmax = min(g_xmin, x.min()), max(g_xmax, x.max())
                            g_ymin, g_ymax = min(g_ymin, y.min()), max(g_ymax, y.max())
                            frame_has_data = True
                            has_any_valid_data = True
                except Exception:
                    pass # 璇诲彇澶辫触涔熺畻浣滅┖鏁版嵁锛屼笉鎶ラ敊

            current_frame_xs.append(data_x)
            current_frame_ys.append(data_y)

        if frame_has_data: 
            frames_x.append(current_frame_xs)
            frames_y.append(current_frame_ys)
            frames_valid_lab.append(f_lab)
        else:
            pass

    if not has_any_valid_data:
        print("[ERROR] No valid data; cannot generate GIF.")
        return

    if xlim is None: xlim = _pad_limits(g_xmin, g_xmax)
    if ylim is None: ylim = _pad_limits(g_ymin, g_ymax)

    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    for c, l in zip(series_colors, series_labels):
        line, = ax.plot([], [], lw=lw, color=c, label=l)
        lines.append(line)
    
    title_obj = ax.set_title("")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if grid: ax.grid(True, alpha=0.3)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.legend(loc="upper left")

    def update(i):
        for j, line in enumerate(lines):
            x_data = frames_x[i][j]
            y_data = frames_y[i][j]
            
            # 濡傛灉璇ョ郴鍒楁湁鏁版嵁锛屽垯鏇存柊
            if x_data is not None and y_data is not None:
                line.set_data(x_data, y_data)
            else:
                line.set_data([], [])
                
        title_obj.set_text(f"{title if title else ''}\nRecord: {frames_valid_lab[i]}")
        return [*lines, title_obj]

    anim = FuncAnimation(fig, update, frames=len(frames_x), interval=int(1000/fps), blit=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer="pillow", fps=fps, dpi=dpi)
    print(f"[OK] GIF Saved: {save_path}")
    if show: plt.show()
    else: plt.close(fig)


# ==============================================================================

if __name__ == "__main__":
    
    MODEL = 'PFSDF'
    
    # =========================================================
    # 浠诲姟 1: Pushover 瀵规瘮 (闈欐€佸浘)
    # =========================================================
    RUN_PUSHOVER = False
    if RUN_PUSHOVER:
        print("\n=== Running Pushover Comparison ===")
        po_temps = ["-20", "0", "20", "40"]
        po_sma_file = "SMA1.out"
        po_paths = [f"Output_data/MC8_{MODEL}/MC8_PO/Pushover/{po_sma_file}" for t in po_temps]
        
        plot_hysteresis_curves_debug(
            po_paths,
            labels=[f"{t}掳C" for t in po_temps],
            title=f"Pushover Comparison - {po_sma_file}",
            # save_path=f"Paint/png/Pushover_{MODEL}_{po_sma_file}.png",
            show=True
        )

    # =========================================================
    # 浠诲姟 2: GIF 鍔ㄧ敾鐢熸垚 (鏀寔鏂囦欢缂哄け鎯呭喌)
    # =========================================================
    RUN_GIF = True
    if RUN_GIF:
        print("\n=== Running GIF Generation ===")

        gif_records = [f"8_{i}" for i in range(1, 81)]

        comparison_config = [
             {"label": "-20", "temp": "-20", "sma": "SMA6_1.out", "color": "blue"},
             {"label": "0",  "temp": "0",  "sma": "SMA6_1.out", "color": "red"},
             {"label": "20",  "temp": "20",  "sma": "SMA6_1.out", "color": "green"},
             {"label": "40",  "temp": "40",  "sma": "SMA6_1.out", "color": "orange"}
        ]
        
        frame_groups = []
        series_labels = [item["label"] for item in comparison_config]
        series_colors = [item.get("color", None) for item in comparison_config]
        path_template = "Output_data/MC8_{model}_{temp}/MC8_IDA_data/{rec}/{sma}"

        for rec in gif_records:
            current_frame_files = []
            for item in comparison_config:
                p = path_template.format(model=MODEL, temp=item["temp"], rec=rec, sma=item["sma"])
                current_frame_files.append(p)
            frame_groups.append(current_frame_files)

        gif_name = f"Compare_{MODEL}_Temperature.gif"
        
        make_hysteresis_gif_from_frames(
            frame_groups,
            save_path=f"Paint/gif/{gif_name}",
            frame_labels=gif_records,
            series_labels=series_labels,
            series_colors=series_colors,
            title=f"IDA Response ({MODEL})",
            fps=2,
            show=False
        )

    # =========================================================
    # =========================================================
    RUN_DEBUG = False
    
    if RUN_DEBUG:
        
        print(f"\n=== Debug Check: {debug_rec} ===")
        debug_paths = []
        debug_path_template = "Output_data/MC8_{model}_{temp}/MC8_IDA_data/{rec}/{sma}"
        
        for item in comparison_config: 
            p = debug_path_template.format(model=MODEL, temp=item["temp"], rec=debug_rec, sma=item["sma"])
            debug_paths.append(p)
            
        plot_hysteresis_curves_debug(
            debug_paths,
            labels=[item["label"] for item in comparison_config],
            title=f"Debug Check: {debug_rec}",
            show=True
        )
