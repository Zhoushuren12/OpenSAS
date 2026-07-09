#!/usr/bin/env python3
"""
Hazard curve interpolation helper (no interaction).

给定结构周期 T_target，在已有的灾害曲线文件夹中插值得到目标周期的 λ(IM) 曲线。
"""

from __future__ import annotations
import pathlib
from pathlib import Path
import numpy as np
base_dir = Path('Output_data')


def parse_period(path: pathlib.Path) -> float:
    try:
        return float(path.stem)
    except ValueError as exc:
        raise ValueError(f"无法从文件名 {path.name} 解析周期") from exc


def read_curve(path: pathlib.Path):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{path} 应为两列数据 (Sa, λ)")
    return data[:, 0], data[:, 1]

def interpolate_curve(folder: pathlib.Path, target_period: float):
    files = sorted(folder.glob("*.txt"), key=parse_period)
    if not files:
        raise FileNotFoundError(f"未找到任何灾害曲线文件：{folder}")

    periods = np.array([parse_period(f) for f in files])

    # 检查目标周期范围
    if target_period < periods.min() or target_period > periods.max():
        raise ValueError(
            f"目标周期 {target_period} s 超出可用范围 "
            f"[{periods.min()}, {periods.max()}] s"
        )

    # 如果目标周期刚好命中现有曲线，直接返回
    matches = np.isclose(periods, target_period)
    if matches.any():
        idx = np.where(matches)[0][0]
        return read_curve(files[idx])

    # 找到包围 T_target 的两条曲线
    hi = np.searchsorted(periods, target_period)
    lo = hi - 1
    T_lo, T_hi = periods[lo], periods[hi]

    sa_lo, lam_lo = read_curve(files[lo])
    sa_hi, lam_hi = read_curve(files[hi])

    # 排序，确保 Sa 单调
    order_lo = np.argsort(sa_lo)
    sa_lo, lam_lo = sa_lo[order_lo], lam_lo[order_lo]

    order_hi = np.argsort(sa_hi)
    sa_hi, lam_hi = sa_hi[order_hi], lam_hi[order_hi]

    # 统一 Sa 网格：用两个曲线的并集（所有出现过的 Sa）
    sa_common = np.union1d(sa_lo, sa_hi)

    # 在各自曲线上插值出公共 Sa 网格上的 λ
    lam_lo_common = np.interp(sa_common, sa_lo, lam_lo)
    lam_hi_common = np.interp(sa_common, sa_hi, lam_hi)

    # 为避免 log 出错，理论上 λ 都 > 0，这里不额外处理
    ln_lam_lo = np.log(lam_lo_common)
    ln_lam_hi = np.log(lam_hi_common)

    # 周期方向在线性插值 ln(λ)
    w = (target_period - T_lo) / (T_hi - T_lo)
    ln_lam_interp = ln_lam_lo + w * (ln_lam_hi - ln_lam_lo)
    lam_interp = np.exp(ln_lam_interp)

    # -----------【新增：log-space 重采样】-----------
    n_resample = 200   # 想要多少点自己改

    log_sa_min = np.log(sa_common.min())
    log_sa_max = np.log(sa_common.max())
    log_sa_new = np.linspace(log_sa_min, log_sa_max, n_resample)
    sa_new = np.exp(log_sa_new)

    # 插值 λ 到新网格（log-space）
    ln_lam_common = np.log(lam_interp)
    ln_lam_new = np.interp(log_sa_new, np.log(sa_common), ln_lam_common)
    lam_new = np.exp(ln_lam_new)

    return sa_new, lam_new


# ===============================
# 🚀 主程序：只需改这里即可
# ===============================
if __name__ == "__main__":

    # ① 修改灾害曲线所在文件夹
    hazard_dir = pathlib.Path("data/Hzarad_curves_LA_Soil-D")

    # ② 修改结构的第一自振周期 T₁（秒）
    T0 = np.loadtxt(base_dir / 'MC8_PFSDF_20' / 'MC8_PO_out' / '周期(s).out')[0]
    T_target = T0    # ← 仅需改这里

    # ③ 修改输出 txt 路径
    out_path = pathlib.Path(f"data/hazard_T{T_target:.3f}.txt")  # ← 可以随便改

    # ====== 插值计算 ======
    sa_vals, lambda_vals = interpolate_curve(hazard_dir, T_target)

    # ====== 写文件 ======
    header = f"# Interpolated hazard curve at T = {T_target:.4f} s"
    data = np.column_stack([sa_vals, lambda_vals])
    np.savetxt(out_path, data, fmt="%.6e", header=header, comments='')

    print(f"已保存到: {out_path}")
