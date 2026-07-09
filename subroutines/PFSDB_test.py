"""PFSDB hysteresis quick-test.

- 基于 SelfCentering 材料模型进行测试
- 包含 k1, k2, sigAct, beta, epsSlip, epsBear, rBear (7个参数)
- 已完全取消 MinMax 位移限制
- 运行 OpenSees 单单元桁架模型并保存滞回曲线
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

# 1. 恢复为你正确的 PFSDB 导入
try:
    from subroutines.PFSDB import PFSDB_pts
except ImportError:
    from PFSDB import PFSDB_pts

DEFAULT_TEMPS = [-20.0, 0.0, 20.0, 40.0]
DEFAULT_STORIES = [1, 2, 3, 4, 5, 6, 7, 8]
# 生成逐渐增大的位移协议
DEFAULT_AMP = [val for x in range(50, 250, 50) for val in (0, x, -x)] + [0]
DEFAULT_DU = 0.5 
OUT_DIR = Path(__file__).resolve().parents[1] / "Output_data" / "PFSDB"

def build_protocol(amp: Sequence[float], du: float) -> np.ndarray:
    """将位移目标点展开为完整的位移历史流"""
    u = np.array([], dtype=float)
    for i in range(len(amp) - 1):
        delta = amp[i + 1] - amp[i]
        steps = max(int(abs(delta) / du), 1)
        segment = np.linspace(amp[i], amp[i + 1], steps, endpoint=False)
        u = np.concatenate([u, segment])
    u = np.append(u, amp[-1])
    return u

# 2. 重构材料创建函数：适配 PFSDB 的 7 个参数
def make_sc_material(pts: Sequence[float], tag: int = 1) -> int:
    """创建 SelfCentering 材料，支持 7 个参数 [k1, k2, sigAct, beta, epsSlip, epsBear, rBear]"""
    if len(pts) < 7:
        raise ValueError(f"pts 长度不足，预期为 7，实际得到 {len(pts)}")
    
    k1, k2, sigAct, beta, epsSlip, epsBear, rBear = map(float, pts[:7])
    sc_tag = tag
    
    # 定义 SelfCentering 材料
    ops.uniaxialMaterial("SelfCentering", sc_tag, k1, k2, sigAct, beta, epsSlip, epsBear, rBear)
    
    return sc_tag

def run_hysteresis(pts: Sequence[float], u_hist: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """执行静态位移加载分析 (已移除 minmax 限制)"""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 1)

    mat_tag = make_sc_material(pts, tag=1)
    ops.element("Truss", 1, 1, 2, 1.0, mat_tag)

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
        forces[i] = -ops.eleForce(1, 1) # 桁架轴力
        ok = ops.analyze(1)
        if ok != 0:
            # 如果不收敛，尝试简单 Newton 以外的算法
            ops.algorithm("NewtonLineSearch")
            ok = ops.analyze(1)
            if ok != 0:
                raise RuntimeError(f"分析在位移 u={us[i]:.4f} 处发散")
            ops.algorithm("Newton")
            
    return us[:-1], forces

def plot_hysteresis(u: np.ndarray, f: np.ndarray, title: str, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(u, f, "-", linewidth=1.2)
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (N)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_combined(traces: List[Tuple[int, np.ndarray, np.ndarray]], title: str, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for story_idx, u, f in traces:
        ax.plot(u, f, "-", label=f"Story {story_idx}", alpha=0.8)
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (N)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_temp_compare(traces: List[Tuple[float, np.ndarray, np.ndarray]], title: str, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for T, u, f in traces:
        ax.plot(u, f, "-", label=f"T={T:.0f}C")
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (N)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PFSDB hysteresis loops.")
    parser.add_argument("-t", "--temps", nargs="+", type=float, default=DEFAULT_TEMPS)
    parser.add_argument("-s", "--stories", nargs="+", type=int, default=DEFAULT_STORIES)
    parser.add_argument("--amp", nargs="+", type=float, default=DEFAULT_AMP)
    parser.add_argument("--du", type=float, default=DEFAULT_DU)
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--combine", action="store_true", default=True)
    parser.add_argument("--temp-compare", action="store_true", default=True)
    args = parser.parse_args()

    temps = sorted(dict.fromkeys(args.temps))
    stories = sorted(dict.fromkeys(args.stories))
    u_hist = build_protocol(args.amp, args.du)

    out_dir: Path = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_compare_data: dict[int, List[Tuple[float, np.ndarray, np.ndarray]]] = {}

    for T in temps:
        # 3.1 调用正确的 PFSDB_pts 接口获取参数 
        pts_all = PFSDB_pts(T=T)
        
        # 3.2 过滤逻辑：寻找长度为 7 的参数列表 (因为你最新发的 PFSDB 代码返回的是 7 个参数)
        pts_only = [p for p in pts_all if isinstance(p, (list, tuple)) and len(p) == 7]
        
        combined_traces: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for story_idx in stories:
            if story_idx < 1 or story_idx > len(pts_only):
                continue
            
            pts = pts_only[story_idx - 1]
            
            print(f"Analyzing: T={T}C, Story={story_idx}...")
            
            # 移除了 minmax 参数传递
            u, f = run_hysteresis(pts, u_hist)
            combined_traces.append((story_idx, u, f))
            temp_compare_data.setdefault(story_idx, []).append((T, u, f))

            name = f"T{T:.0f}_story{story_idx}"
            plot_hysteresis(u, f, f"PFSDB Hysteresis - {name}", out_dir / f"hyst_{name}.png")

        if args.combine and combined_traces:
            plot_combined(combined_traces, f"Stories Comparison at T={T:.0f}C", out_dir / f"hyst_T{T:.0f}_combined.png")

    if args.temp_compare:
        for s_idx, traces in temp_compare_data.items():
            plot_temp_compare(traces, f"Temp Comparison - Story {s_idx}", out_dir / f"hyst_story{s_idx}_temp_compare.png")

if __name__ == "__main__":
    main()