"""SMABD hysteresis quick-test.

- Choose temperature(s) and story index(es)
- Define your own displacement protocol (amp list + step du)
- Runs an OpenSees twoNodeLink with SMABD_pts (SelfCentering with Bearing hardening)
- Saves hysteresis plots and raw data under Output_data/SMABD

Example:
    python subroutines/SMAB_test.py -t -20 0 20 40 -s 1 4 \
        --amp 0 20 -20 30 -30 0 --du 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

# 1. 导入新的 SMABD 模型接口
try:
    from subroutines.SMADB import SMADB_pts
except ImportError:
    from SMADB import SMADB_pts


DEFAULT_TEMPS = [-20.0, 0.0, 20.0, 40.0]
DEFAULT_STORIES = [1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_AMP = [0, 100, 0, -100, 0, 150, 0, -150, 0, 200, 0, -200, 0]
DEFAULT_DU = 0.5  # displacement increment
OUT_DIR = Path(__file__).resolve().parents[1] / "Output_data" / "SMADB"


def build_protocol(amp: Sequence[float], du: float) -> np.ndarray:
    """Expand piecewise-linear targets into a full displacement history."""
    u = np.array([], dtype=float)
    for i in range(len(amp) - 1):
        delta = amp[i + 1] - amp[i]
        steps = max(int(abs(delta) / du), 1)
        segment = np.linspace(amp[i], amp[i + 1], steps, endpoint=False)
        u = np.concatenate([u, segment])
    u = np.append(u, amp[-1])
    return u


def make_smabd_material(pts: Sequence[float], minmax: Optional[Sequence[float]] = None, tag: int = 1) -> int:
    """Create SelfCentering (with optional MinMax wrapper) and return the tag to use in the element."""
    # 2. 核心修改：提取全部 7 个参数 [k1, k2, sigAct, beta, slip_disp, Dm, Km]
    if len(pts) < 7:
        raise ValueError(f"pts 长度不足，预期为 7 (包含硬化参数)，实际得到 {len(pts)}")
    
    # 传入 7 个参数激活 Bearing 硬化机制
    ops.uniaxialMaterial("SelfCentering", tag, *map(float, pts[:7]))
    mat_tag = tag
    
    if minmax is not None:
        minmax_tag = tag * 10
        ops.uniaxialMaterial("MinMax", minmax_tag, tag, "-min", float(minmax[0]), "-max", float(minmax[1]))
        mat_tag = minmax_tag
        
    return mat_tag


def run_hysteresis(pts: Sequence[float], u_hist: Sequence[float], minmax: Optional[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Run static analysis for a given displacement history; returns (u, F)."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 1)

    mat_tag = make_smabd_material(pts, minmax=minmax, tag=1)
    ops.element("twoNodeLink", 1, 1, 2, "-mat", mat_tag, "-dir", 1)

    us = np.append(np.asarray(u_hist, dtype=float), 0.0)
    ops.timeSeries("Path", 1, "-dt", 1.0, "-values", *us)
    ops.pattern("Plain", 1, 1)
    ops.sp(2, 1, 1)

    ops.constraints("Lagrange")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("EnergyIncr", 1e-6, 50)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")

    N = len(us)
    forces = np.zeros(N - 1)
    for i in range(N - 1):
        forces[i] = -ops.eleForce(1, 1)
        ok = ops.analyze(1)
        if ok != 0:
            # 引入一个简单的回退机制，当 Newton 发散时尝试 NewtonLineSearch
            ops.algorithm("NewtonLineSearch")
            ok = ops.analyze(1)
            if ok != 0:
                raise RuntimeError(f"Analysis failed at step {i} (u={us[i]:.4f})")
            ops.algorithm("Newton") # 恢复 Newton
            
    return us[:-1], forces


def plot_hysteresis(u: np.ndarray, f: np.ndarray, title: str, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(u, f, "-", linewidth=1.5)
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (N)")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined(traces: List[Tuple[int, np.ndarray, np.ndarray]], title: str, outfile: Path) -> None:
    """Plot multiple story curves on one figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for story_idx, u, f in traces:
        ax.plot(u, f, "-", linewidth=1.2, label=f"Story {story_idx}")
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (N)")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_temp_compare(traces: List[Tuple[float, np.ndarray, np.ndarray]], title: str, outfile: Path) -> None:
    """Plot multiple temperature curves for a single story."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for T, u, f in traces:
        ax.plot(u, f, "-", linewidth=1.2, label=f"T={T:.0f}C")
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (N)")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SMABD hysteresis loops with custom protocol.")
    parser.add_argument("-t", "--temps", nargs="+", type=float, default=DEFAULT_TEMPS, help="Temperatures (degC).")
    parser.add_argument(
        "-s", "--stories", nargs="+", type=int, default=DEFAULT_STORIES, help="Story indices (1-based)."
    )
    parser.add_argument("--amp", nargs="+", type=float, default=DEFAULT_AMP, help="Target displacement points.")
    parser.add_argument("--du", type=float, default=DEFAULT_DU, help="Displacement increment between targets.")
    parser.add_argument("--outdir", type=Path, default=OUT_DIR, help="Output folder for plots/data.")
    parser.add_argument("--save-data", action="store_true", help="Also save u,F to CSV.")
    parser.add_argument(
        "--combine",
        action="store_true",
        default=True,
        help="Plot all chosen stories on one figure per temperature (default: on)."
    )
    parser.add_argument(
        "--temp-compare",
        action="store_true",
        default=True,
        help="Plot temperature comparison for each chosen story (one figure per story, default: on)."
    )
    args = parser.parse_args()

    temps = sorted(dict.fromkeys(args.temps))
    stories = sorted(dict.fromkeys(args.stories))
    u_hist = build_protocol(args.amp, args.du)

    out_dir: Path = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_compare_data: dict[int, List[Tuple[float, np.ndarray, np.ndarray]]] = {}

    for T in temps:
        # 3. 调用新接口 SMABD_pts
        bundle = SMADB_pts(T=T)
        if len(bundle) < 3:
            raise ValueError("SMADB_pts must return pts per story followed by minmax1 and minmax2.")
            
        pts_only = bundle[:-2]
        minmax1, minmax2 = bundle[-2], bundle[-1]
        
        combined_traces: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for story_idx in stories:
            if story_idx < 1 or story_idx > len(pts_only):
                raise IndexError(f"Story index {story_idx} out of range (1..{len(pts_only)})")
                
            pts = pts_only[story_idx - 1]
            minmax = minmax1 if story_idx == 1 else minmax2

            u, f = run_hysteresis(pts, u_hist, minmax=minmax)
            combined_traces.append((story_idx, u, f))
            temp_compare_data.setdefault(story_idx, []).append((T, u, f))

            name = f"T{T:.0f}_story{story_idx}"
            png_path = out_dir / f"hyst_{name}.png"
            plot_hysteresis(u, f, f"SMABD Hysteresis - {name}", png_path)

            if args.save_data:
                data_path = out_dir / f"hyst_{name}.csv"
                np.savetxt(data_path, np.column_stack([u, f]), delimiter=",", header="u,F", comments="")

            print(f"[done] T={T:.1f} story={story_idx} -> {png_path}")

        if args.combine and combined_traces:
            name = f"T{T:.0f}_stories_all"
            png_path = out_dir / f"hyst_{name}.png"
            plot_combined(combined_traces, f"Stories Comparison at T={T:.0f}C", png_path)
            if args.save_data:
                rows = []
                for story_idx, u, f in combined_traces:
                    rows.append(np.column_stack([np.full_like(u, story_idx, dtype=float), u, f]))
                data = np.vstack(rows)
                data_path = out_dir / f"hyst_{name}.csv"
                np.savetxt(data_path, data, delimiter=",", header="story,u,F", comments="")
            print(f"[done] combined plot -> {png_path}")

    if args.temp_compare and temp_compare_data:
        for story_idx, traces in temp_compare_data.items():
            traces_sorted = sorted(traces, key=lambda x: x[0])
            name = f"story{story_idx}_temps_all"
            png_path = out_dir / f"hyst_{name}.png"
            plot_temp_compare(traces_sorted, f"Temperature Comparison - Story {story_idx}", png_path)
            if args.save_data:
                rows = []
                for T, u, f in traces_sorted:
                    rows.append(np.column_stack([np.full_like(u, T, dtype=float), u, f]))
                data = np.vstack(rows)
                data_path = out_dir / f"hyst_{name}.csv"
                np.savetxt(data_path, data, delimiter=",", header="temp,u,F", comments="")
            print(f"[done] temp compare plot -> {png_path}")


if __name__ == "__main__":
    main()