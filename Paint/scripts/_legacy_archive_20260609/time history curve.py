from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def plot_idr_timehistory(
    case_dirs,
    labels=None,
    *,
    time_file="time.out",
    top_disp_file="Disp7.out",
    bot_disp_file="Disp6.out",
    sdr_file=None,
    time_col=0,
    disp_col=None,
    sdr_col=None,
    story_height=4300.0,
    sdr_scale=100.0,
    dt=0.02,
    title=None,
    xlabel="Time (s)",
    ylabel="Inter-story drift ratio (%)",
    figsize=(12, 7),
    lw=1.6,
    grid=True,
    xlim=None,
    ylim=None,
    # ===== inset =====
    use_inset=True,
    inset_bbox=(0.30, 0.68, 0.25, 0.25),
    inset_xlim=(8.0, 8.4),
    inset_ylim=(4.5, 5.25),
    peak_offset_map=None,       # dict: label -> dy
    # ===== save / show =====
    save_path=None,
    show=True,
):
    case_dirs = [Path(p) for p in case_dirs]
    if labels is None:
        labels = [p.name for p in case_dirs]

    plt.rc("font", family="Times New Roman")
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=figsize)

    axins = None
    if use_inset:
        axins = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=inset_bbox,
            bbox_transform=ax.transAxes,
            loc="lower left",
        )

    def _pick_col(arr, col):
        if arr.ndim == 2:
            return arr[:, col] if col is not None else arr[:, -1]
        return arr

    def _pick_item(value, idx):
        if isinstance(value, (list, tuple)):
            if len(value) != len(case_dirs):
                raise ValueError("List-like file args must have same length as case_dirs.")
            return value[idx]
        return value

    for idx, (d, lab) in enumerate(zip(case_dirs, labels)):
        time_name = _pick_item(time_file, idx)
        top_name = _pick_item(top_disp_file, idx)
        bot_name = _pick_item(bot_disp_file, idx)
        sdr_name = _pick_item(sdr_file, idx)

        time_path = d / time_name if time_name else None
        top_path  = d / top_name if top_name else None
        bot_path  = d / bot_name if bot_name else None
        sdr_path  = d / sdr_name if sdr_name else None

        # ---- 鎵句笉鍒版枃浠跺氨璺宠繃 ----
        if sdr_name:
            if not (sdr_path and sdr_path.exists()):
                continue
        else:
            if not (top_path.exists() and bot_path.exists()):
                continue

        t = None
        if time_path and time_path.exists() and time_path.stat().st_size > 0:
            t_raw = np.loadtxt(time_path)
            t = _pick_col(t_raw, time_col)

        if sdr_name:
            sdr = _pick_col(np.loadtxt(sdr_path), sdr_col)
            idr = sdr * sdr_scale
            n = len(idr)
        else:
            disp_t = _pick_col(np.loadtxt(top_path), disp_col)
            disp_b = _pick_col(np.loadtxt(bot_path), disp_col)
            n = min(len(disp_t), len(disp_b))
            idr = (disp_t[:n] - disp_b[:n]) / story_height * 100.0

        if t is None:
            t = np.arange(n) * dt if dt is not None else np.arange(n)
        else:
            n = min(n, len(t))
            t = t[:n]
            idr = idr[:n]

        # 涓诲浘
        ax.plot(t, idr, label=lab, linewidth=lw)

        # inset
        if use_inset:
            line = axins.plot(t, idr, linewidth=1.0)
            col = line[0].get_color()

            idx = np.argmax(idr)
            dy = 0.075
            if peak_offset_map is not None:
                dy = peak_offset_map.get(lab, dy)

            axins.annotate(
                f"{idr[idx]:.2f}%",
                xy=(t[idx], idr[idx]),
                xytext=(t[idx] + 0.1, idr[idx] + dy),
                arrowprops=dict(
                    shrink=0.05,
                    width=1,
                    headwidth=6,
                    color=col,
                ),
                fontsize=13,
                color=col,
            )

    # ===== 涓诲浘鏍煎紡 =====
    if title:
        ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", direction="in", labelsize=14)

    if grid:
        ax.grid(True, linestyle="--", alpha=0.3)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    ax.legend(fontsize=14)

    # ===== inset 鏍煎紡 =====
    if use_inset:
        axins.set_xlim(*inset_xlim)
        axins.set_ylim(*inset_ylim)
        axins.tick_params(
            axis="both",
            which="both",
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False
        )
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ===================== IDA 鍒嗘瀽 =====================

# 可选:
# "temperature" -> 同一楼层对比不同温度
# "story"       -> 同一温度对比不同楼层
COMPARE_MODE = "temperature"

MODEL = "PFSDF"
RECORD_RANGE = range(1, 60)
RECORD_TEMPLATE = "2_{i}"
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "Output_data"

# mode = "temperature"
TEMPERATURES = [-20, 0, 20, 40]
COMPARE_STORY = 2

# mode = "story"
TARGET_TEMPERATURE = 20
STORIES_TO_COMPARE = [2, 3, 4, 5, 6, 7]

if COMPARE_MODE not in {"temperature", "story"}:
    raise ValueError("COMPARE_MODE must be 'temperature' or 'story'.")
if COMPARE_MODE == "temperature" and COMPARE_STORY < 2:
    raise ValueError("COMPARE_STORY must be >= 2 when using Disp{n} - Disp{n-1}.")
if COMPARE_MODE == "story" and any(story < 2 for story in STORIES_TO_COMPARE):
    raise ValueError("All STORIES_TO_COMPARE entries must be >= 2.")

for i in RECORD_RANGE:
    record = RECORD_TEMPLATE.format(i=i)

    if COMPARE_MODE == "temperature":
        case_dirs = [
            OUTPUT_ROOT / f"MC8_{MODEL}_{temp}" / "MC8_IDA_data" / record
            for temp in TEMPERATURES
        ]
        labels = [f"{temp}C" for temp in TEMPERATURES]
        top_disp_file = f"Disp{COMPARE_STORY}.out"
        bot_disp_file = f"Disp{COMPARE_STORY - 1}.out"
        title = f"IDA time history ({record}) Story {COMPARE_STORY - 1}-{COMPARE_STORY}"
        peak_offset_map = {"-20C": 0.13, "0C": 0.075, "20C": 0.075, "40C": 0.075}
    else:
        base_case_dir = OUTPUT_ROOT / f"MC8_{MODEL}_{TARGET_TEMPERATURE}" / "MC8_IDA_data" / record
        case_dirs = [base_case_dir for _ in STORIES_TO_COMPARE]
        labels = [f"Story {story - 1}-{story}" for story in STORIES_TO_COMPARE]
        top_disp_file = [f"Disp{story}.out" for story in STORIES_TO_COMPARE]
        bot_disp_file = [f"Disp{story - 1}.out" for story in STORIES_TO_COMPARE]
        title = f"IDA time history ({record}) {TARGET_TEMPERATURE}C"
        peak_offset_map = None

    plot_idr_timehistory(
        case_dirs,
        labels=labels,
        title=title,
        top_disp_file=top_disp_file,
        bot_disp_file=bot_disp_file,
        story_height=4300.0,
        use_inset=False,
        inset_xlim=(8.0, 8.4),
        inset_ylim=(4.5, 5.25),
        peak_offset_map=peak_offset_map,
    )

# ===================== 鏃剁▼鍒嗘瀽 =====================

# for i in range(1,45):

#     record = f"{i}"
#     model = "PFSDF"
#     Level = "ERE"
#     case_dirs = [
#         f"E:\Opensees-PFSDF-Temp\Output_data\MC8_{model}_-20\MC8_TH_{Level}_data\{record}",
#         f"E:\Opensees-PFSDF-Temp\Output_data\MC8_{model}_0\MC8_TH_{Level}_data\{record}",
#         f"E:\Opensees-PFSDF-Temp\Output_data\MC8_{model}_20\MC8_TH_{Level}_data\{record}",
#         f"E:\Opensees-PFSDF-Temp\Output_data\MC8_{model}_40\MC8_TH_{Level}_data\{record}",

#     ]

#     plot_idr_timehistory(
#         case_dirs,
#         labels=["-20C", "0C", "20C", "40C"],
#         title=f'IDA time history ({record})',
#         sdr_file="SDR7.out",
#         use_inset=False,
#         inset_xlim=(8.0, 8.4),
#         inset_ylim=(4.5, 5.25),
#         peak_offset_map={"-20C": 0.13, "0C": 0.075, "20C": 0.075, "40C": 0.075},
#     )

