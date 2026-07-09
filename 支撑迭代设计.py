# -*- coding: utf-8 -*-
"""
scanA_then_refine_sharedA_PFSDF_vs_SMABF.py

用途：
- 先做 A 的“粗扫”（grid/coarse scan），快速判断 gap(A)=median_SMABF-median_PFSDF 的可达范围；
- 再围绕粗扫得到的“最佳 gap 点”做“细扫”（local refine），找到更优的 A；
- 同时始终满足 MCE mean-limit 约束（按 MCE_LIMIT_TARGET 指定约束对象）。

约束（MCE）：
  mean( max_story_IDR over MCE records ) <= IDR_LIMIT

目标（ERE）：
  gap(A) = median_SMABF(A) - median_PFSDF(A)  越大越好
  并检查是否达到阈值：gap(A) >= DELTA_MEDIAN_MIN

输出：
- CSV 日志：subroutines/扫描日志_scanA_then_refine.csv
- 控制台打印：每个 A 的 MCE 可行性、ERE gap、当前最优点

注意：
- 这是“扫描法”，不依赖 gap 的单调性，适合你这种“非单调 + 噪声”的情况。
- 成本高：每个 A 至少跑 1 次 MCE（约束对象需要的模型）+ 2 次 ERE（两模型）。
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json, csv, shutil, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np


# ===================== 用户需要改的地方 =====================
MODEL_P = "MC8_PFSDF"
MODEL_S = "MC8_SMABF"

ERE_SPEC = Path(r"Spectrum\\ERE Level Spectrum.txt")
MCE_SPEC = Path(r"Spectrum\\MCE Level Spectrum.txt")

RECORD_IDS_ERE = [f"{i}" for i in range(1, 45)]
RECORD_IDS_MCE = [f"{i}" for i in range(11, 22)]

# MCE mean-limit 施加对象：
#   "P"    : 只约束 PFSDF
#   "S"    : 只约束 SMABF
#   "BOTH" : 两者都约束
MCE_LIMIT_TARGET = "S"

IDR_LIMIT = 0.0375
DELTA_MEDIAN_MIN = 0.003

PARALLEL = 6

# --------- A 扫描区间与分辨率（按你的算力调整） ----------
# 粗扫（建议 8~15 个点）
A_COARSE_LIST = [ 800, 1200, 1600, 2000, 2500, 3000, 3500, 4000]

# 细扫：围绕粗扫最优点 best_A，做 [best_A - span, best_A + span] 的等间距扫描
REFINE_SPAN = 200       # 半宽
REFINE_POINTS = 9         # 细扫点数（奇数更好，包含中心点）

# 如果你想：只在“粗扫已可行”时才细扫
REFINE_ONLY_IF_FEASIBLE = True

# ==========================================================


# ===================== 全局工程几何/参数 =====================
ROOT_P = Path("Output_data") / MODEL_P
ROOT_S = Path("Output_data") / MODEL_S

OUT_P  = ROOT_P / "MC8_TH_design_data"
POST_P = ROOT_P / "MC8_TH_design_data_out"

OUT_S  = ROOT_S / "MC8_TH_design_data"
POST_S = ROOT_S / "MC8_TH_design_data_out"

HEIGHTS_MM     = [5500, 4300, 4300, 4300, 4300, 4300, 4300, 4300]
N_STORY        = len(HEIGHTS_MM)
N_BAY          = 3
L_BAY_EDGE_MM  = 9150

L_TEST_MM     = 450.0
THETA_TARGET  = 0.03
EPS_TARGET    = 0.06

LIMIT_RATIO   = 0.12
COEFFS_INIT   =  [1.0,0.7906,0.7264,0.6619,0.5876,0.5249,0.4383,0.2398]
# COEFFS_INIT   = [1.0000, 1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000]

SUBROUTINES_DIR = Path("subroutines")
CFG_FILE        = SUBROUTINES_DIR / "sma_iter_config.json"
LOG_FILE        = SUBROUTINES_DIR / "扫描日志_scanA_then_refine.csv"
# ==========================================================


# ===================== 工具函数 =====================
def ensure_dir_for_file(fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)

def validate_target(tag: str):
    if tag not in ("P", "S", "BOTH"):
        raise ValueError(f"Unknown MCE_LIMIT_TARGET={tag!r}. Use 'P', 'S', or 'BOTH'.")

def _alpha(h_clear_mm: float, l_bay_edge_mm: float) -> float:
    return math.atan2(h_clear_mm, l_bay_edge_mm / 2.0)

def calculate_effective_lengths(
    heights_mm, l_bay_edge_mm, theta_target, eps_target
) -> List[float]:
    L_effs = []
    for h in heights_mm:
        a = _alpha(h, l_bay_edge_mm)
        delta_story = theta_target * h
        delta_sma   = delta_story * math.cos(a)
        L_eff_i     = max(delta_sma / eps_target, 1e-6)
        L_effs.append(float(L_eff_i))
    return L_effs

def write_iter_cfg(A_base: float, coeffs: List[float], limit_ratio: float):
    L_effs = calculate_effective_lengths(
        HEIGHTS_MM, L_BAY_EDGE_MM, THETA_TARGET, EPS_TARGET
    )
    L1 = float(L_effs[0])
    L2 = float(np.median(L_effs[1:])) if len(L_effs) > 1 else L1

    minmax1 = [-limit_ratio * L1,  limit_ratio * L1]
    minmax2 = [-limit_ratio * L2,  limit_ratio * L2]

    cfg = {
        "A_base_mm2": float(A_base),
        "A": float(A_base),
        "coeffs": coeffs,
        "limit_ratio": float(limit_ratio),
        "L_bay_edge_mm": float(L_BAY_EDGE_MM),
        "heights_mm": HEIGHTS_MM,
        "L_test_mm": float(L_TEST_MM),
        "theta_target": float(THETA_TARGET),
        "eps_target": float(EPS_TARGET),
        "L1": float(L1), "L2": float(L2),
        "minmax1": minmax1, "minmax2": minmax2
    }
    ensure_dir_for_file(CFG_FILE)
    CFG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

def read_idr_max_per_record(post_dir: Path, record_ids: List[str]) -> np.ndarray:
    per_record_max = []
    for rec in record_ids:
        f = post_dir / f"{rec}" / "层间位移角.out"
        arr = np.loadtxt(f, dtype=float).reshape(-1)
        story_vals = arr[1:N_STORY + 1]  # 去掉底部 0，保留全部楼层
        per_record_max.append(float(story_vals.max()))
    return np.asarray(per_record_max, dtype=float)

def stats_from_post(post_dir: Path, record_ids: List[str]) -> Dict[str, float]:
    idr = read_idr_max_per_record(post_dir, record_ids)
    return {
        "mean": float(np.mean(idr)),
        "median": float(np.median(idr)),
        "max": float(np.max(idr)),
    }

def run_once(
    model_name: str,
    output_root: Path,
    out_dir: Path,
    post_dir: Path,
    A_base: float,
    coeffs: List[float],
    limit_ratio: float,
    spectrum_path: Path,
    level_tag: str,
    record_ids: List[str],
):
    write_iter_cfg(A_base, coeffs, limit_ratio)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    if post_dir.exists():
        shutil.rmtree(post_dir)

    from MRFcore.MRF import MRF
    from MRFcore.DataProcessing import DataProcessing

    note = f"[scanA | {model_name} | {level_tag}×{len(record_ids)}] A_base={A_base:.0f}"
    model = MRF(model_name, Nstory=N_STORY, Nbay=N_BAY, heights=HEIGHTS_MM, notes=note, script="py")
    model.select_ground_motions(record_ids, suffix=".txt")

    # T1 = np.loadtxt(output_root / "MC8_PO_out" / "周期(s).out")[0]
    T1 =1.5
    model.scale_ground_motions(
        method="c",
        para=(0.2 * T1, 1.5 * T1),
        path_spec_code=spectrum_path,
        save_SF=True,
        plot=False,
    )

    model.set_running_parameters(
        Output_dir=out_dir, fv_duration=30,
        display=True, auto_quit=True, folder_exists="overwrite"
    )

    try:
        model.run_time_history(print_result=False, parallel=PARALLEL)
    except Exception as e:
        print("并行失败，改串行：", e)
        model.run_time_history(print_result=False, parallel=1)

    dp = DataProcessing(out_dir)
    dp.set_output_dir(post_dir, cover=1)
    dp.read_results(
        "mode","IDR","CIDR","PFA","PFV","shear","panelZone","beamHinge","columnHinge",
        print_result=False
    )
    dp.read_th()

def compute_mce_feasibility(A: float, coeffs: List[float]) -> Tuple[bool, float, Dict[str, float], Dict[str, float]]:
    """
    返回：
      ok_mce, ratio_used, mce_p, mce_s
    ratio_used 根据 MCE_LIMIT_TARGET：
      P   -> mean_P/limit
      S   -> mean_S/limit
      BOTH-> max(mean_P, mean_S)/limit
    """
    need_p = (MCE_LIMIT_TARGET in ("P", "BOTH"))
    need_s = (MCE_LIMIT_TARGET in ("S", "BOTH"))

    mce_p = {"mean": np.nan, "median": np.nan, "max": np.nan}
    mce_s = {"mean": np.nan, "median": np.nan, "max": np.nan}

    if need_p:
        run_once(MODEL_P, ROOT_P, OUT_P, POST_P, A, coeffs, LIMIT_RATIO, MCE_SPEC, "MCE", RECORD_IDS_MCE)
        mce_p = stats_from_post(POST_P, RECORD_IDS_MCE)
    if need_s:
        run_once(MODEL_S, ROOT_S, OUT_S, POST_S, A, coeffs, LIMIT_RATIO, MCE_SPEC, "MCE", RECORD_IDS_MCE)
        mce_s = stats_from_post(POST_S, RECORD_IDS_MCE)

    if MCE_LIMIT_TARGET == "P":
        ok = (mce_p["mean"] <= IDR_LIMIT)
        ratio = mce_p["mean"] / IDR_LIMIT
    elif MCE_LIMIT_TARGET == "S":
        ok = (mce_s["mean"] <= IDR_LIMIT)
        ratio = mce_s["mean"] / IDR_LIMIT
    else:
        ok = (mce_p["mean"] <= IDR_LIMIT) and (mce_s["mean"] <= IDR_LIMIT)
        ratio = max(mce_p["mean"], mce_s["mean"]) / IDR_LIMIT

    return bool(ok), float(ratio), mce_p, mce_s

def compute_ere_gap(A: float, coeffs: List[float]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    返回 gap(S-P) 以及 ere_p/ere_s 统计
    """
    run_once(MODEL_P, ROOT_P, OUT_P, POST_P, A, coeffs, LIMIT_RATIO, ERE_SPEC, "ERE", RECORD_IDS_ERE)
    ere_p = stats_from_post(POST_P, RECORD_IDS_ERE)

    run_once(MODEL_S, ROOT_S, OUT_S, POST_S, A, coeffs, LIMIT_RATIO, ERE_SPEC, "ERE", RECORD_IDS_ERE)
    ere_s = stats_from_post(POST_S, RECORD_IDS_ERE)

    gap = float(ere_s["median"] - ere_p["median"])
    return gap, ere_p, ere_s

def init_log():
    ensure_dir_for_file(LOG_FILE)
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "phase", "A",
            "MCE_ok", "MCE_ratio_used", "MCE_limit_target",
            "MCE_mean_P", "MCE_median_P", "MCE_max_P",
            "MCE_mean_S", "MCE_median_S", "MCE_max_S",
            "ERE_median_P", "ERE_mean_P", "ERE_max_P",
            "ERE_median_S", "ERE_mean_S", "ERE_max_S",
            "ERE_gap(S-P)", "ERE_ok(delta)",
            "best_A_so_far", "best_gap_so_far",
            "note"
        ])

def log_row(
    phase: str, A: float,
    ok_mce: bool, ratio_used: float,
    mce_p: Dict[str, float], mce_s: Dict[str, float],
    ere_p: Optional[Dict[str, float]], ere_s: Optional[Dict[str, float]],
    gap: Optional[float],
    best_A: float, best_gap: float,
    note: str
):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if ere_p is None:
            ere_p = {"mean": np.nan, "median": np.nan, "max": np.nan}
        if ere_s is None:
            ere_s = {"mean": np.nan, "median": np.nan, "max": np.nan}
        if gap is None:
            gap = np.nan

        ok_ere = int((gap >= DELTA_MEDIAN_MIN) if np.isfinite(gap) else 0)

        w.writerow([
            phase, f"{A:.6f}",
            int(ok_mce), f"{ratio_used:.6f}", MCE_LIMIT_TARGET,
            f"{mce_p['mean']:.6f}" if np.isfinite(mce_p["mean"]) else "",
            f"{mce_p['median']:.6f}" if np.isfinite(mce_p["median"]) else "",
            f"{mce_p['max']:.6f}" if np.isfinite(mce_p["max"]) else "",
            f"{mce_s['mean']:.6f}" if np.isfinite(mce_s["mean"]) else "",
            f"{mce_s['median']:.6f}" if np.isfinite(mce_s["median"]) else "",
            f"{mce_s['max']:.6f}" if np.isfinite(mce_s["max"]) else "",
            f"{ere_p['median']:.6f}" if np.isfinite(ere_p["median"]) else "",
            f"{ere_p['mean']:.6f}" if np.isfinite(ere_p["mean"]) else "",
            f"{ere_p['max']:.6f}" if np.isfinite(ere_p["max"]) else "",
            f"{ere_s['median']:.6f}" if np.isfinite(ere_s["median"]) else "",
            f"{ere_s['mean']:.6f}" if np.isfinite(ere_s["mean"]) else "",
            f"{ere_s['max']:.6f}" if np.isfinite(ere_s["max"]) else "",
            f"{gap:+.6f}" if np.isfinite(gap) else "",
            ok_ere,
            f"{best_A:.6f}", f"{best_gap:+.6f}",
            note
        ])

def linspace_int(center: float, span: float, n: int) -> List[float]:
    if n <= 1:
        return [float(center)]
    lo = center - span
    hi = center + span
    xs = np.linspace(lo, hi, n)
    return [float(x) for x in xs]
# ============================================================


def evaluate_A(A: float, coeffs: List[float], phase: str,
               best_A: float, best_gap: float) -> Tuple[bool, float, float, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], float, bool, float, float]:
    """
    评估单个 A：
      - 先算 MCE 可行性（按 target）
      - 若可行：算 ERE gap
    返回：
      ok_mce, ratio_used, gap, mce_p, mce_s, ere_p, ere_s, gap, ok_ere, new_best_A, new_best_gap
    """
    print(f"\n[{phase}] === A={A:.2f} ===")
    ok_mce, ratio_used, mce_p, mce_s = compute_mce_feasibility(A, coeffs)
    print(f"[MCE] ok={ok_mce} | ratio_used={ratio_used:.3f} | limit={IDR_LIMIT:.6f} | target={MCE_LIMIT_TARGET}")
    print(f"      mean_P={mce_p['mean'] if np.isfinite(mce_p['mean']) else np.nan:.6f} | mean_S={mce_s['mean'] if np.isfinite(mce_s['mean']) else np.nan:.6f}")

    if not ok_mce:
        note = "MCE infeasible -> skip ERE"
        log_row(phase, A, ok_mce, ratio_used, mce_p, mce_s, None, None, None, best_A, best_gap, note)
        return ok_mce, ratio_used, np.nan, mce_p, mce_s, {"mean": np.nan, "median": np.nan, "max": np.nan}, {"mean": np.nan, "median": np.nan, "max": np.nan}, np.nan, False, best_A, best_gap

    gap, ere_p, ere_s = compute_ere_gap(A, coeffs)
    ok_ere = (gap >= DELTA_MEDIAN_MIN)
    print(f"[ERE] median_P={ere_p['median']:.6f} | median_S={ere_s['median']:.6f} | gap(S-P)={gap:+.6f} | ok(delta)={ok_ere}")

    # best 更新（只在可行域内比较 gap）
    if np.isfinite(gap) and gap > best_gap + 1e-12:
        best_gap = gap
        best_A = A
        note = f"update best -> gap={best_gap:+.6f}"
    else:
        note = "feasible"

    log_row(phase, A, ok_mce, ratio_used, mce_p, mce_s, ere_p, ere_s, gap, best_A, best_gap, note)
    return ok_mce, ratio_used, gap, mce_p, mce_s, ere_p, ere_s, gap, ok_ere, best_A, best_gap


# ===================== 主程序 =====================
if __name__ == "__main__":
    validate_target(MCE_LIMIT_TARGET)
    init_log()

    coeffs = COEFFS_INIT.copy()

    best_A = float("nan")
    best_gap = -1e99
    best_ok_ere = False

    # ---------------------- 1) 粗扫 ----------------------
    print("\n==================== COARSE SCAN ====================")
    for A in A_COARSE_LIST:
        ok_mce, ratio_used, gap, *_rest, ok_ere, best_A, best_gap = evaluate_A(
            float(A), coeffs, "COARSE", best_A if np.isfinite(best_A) else float(A), best_gap
        )
        if ok_ere:
            best_ok_ere = True

    print("\n[COARSE] best_so_far:", f"A={best_A:.2f}, best_gap={best_gap:+.6f}, target={DELTA_MEDIAN_MIN:.6f}")
    if best_gap < -1e90:
        print(">>> 没有任何 A 满足 MCE 可行域（请扩大 A_COARSE_LIST 或放宽 MCE 约束）")
        raise SystemExit(1)

    # 是否细扫
    if REFINE_ONLY_IF_FEASIBLE and not np.isfinite(best_A):
        print(">>> best_A 不存在，跳过细扫。")
        raise SystemExit(0)

    # ---------------------- 2) 细扫 ----------------------
    print("\n==================== REFINE SCAN ====================")
    refine_list = linspace_int(best_A, REFINE_SPAN, REFINE_POINTS)

    # 去重 + 排序
    refine_list = sorted(list({float(x) for x in refine_list if x > 0}))

    for A in refine_list:
        ok_mce, ratio_used, gap, *_rest, ok_ere, best_A, best_gap = evaluate_A(
            float(A), coeffs, "REFINE", best_A, best_gap
        )
        if ok_ere:
            best_ok_ere = True

    print("\n==================== SUMMARY ====================")
    print(f"best_A={best_A:.2f}")
    print(f"best_gap={best_gap:+.6f}  (target >= {DELTA_MEDIAN_MIN:.6f})")
    if best_ok_ere:
        print(">>> ✅ 至少有一个 A 达到目标阈值（gap>=DELTA）")
    else:
        print(">>> ⚠️ 扫描范围内未达到目标阈值（gap>=DELTA）")
        print("建议：扩大 A_COARSE_LIST 范围或改动设计自由度（不仅仅调 A），或重新检查目标定义/谱/记录集。")
