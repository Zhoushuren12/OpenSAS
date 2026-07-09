import openseespy.opensees as ops
from typing import Optional, List, Tuple, Dict
import numpy as np
import json
import math
from pathlib import Path

# ==============================
# 0) 配置与通用工具
# ==============================
_CFG_CACHE = None
_CFG_MTIME = None

def _load_iter_cfg():
    global _CFG_CACHE, _CFG_MTIME
    here = Path(__file__).resolve().parent
    cfg_path = here / "sma_iter_config.json"

    if not cfg_path.exists():
        cfg_path = here.parent / "subroutines" / "sma_iter_config.json"
        if not cfg_path.exists():
            return None

    stat = cfg_path.stat()
    if _CFG_CACHE is None or _CFG_MTIME != stat.st_mtime:
        try:
            _CFG_CACHE = json.loads(cfg_path.read_text(encoding="utf-8"))
            _CFG_MTIME = stat.st_mtime
        except:
            return None
    return _CFG_CACHE

def _alpha(h_clear_mm: float, l_bay_edge_mm: float) -> float:
    return math.atan2(h_clear_mm, l_bay_edge_mm / 2.0)

def _effective_lengths(heights_mm, l_bay_edge_mm, theta_target, eps_target, L_test_mm):
    # 目前你这版没用到，可保留备用
    if eps_target == 0:
        return [1000.0] * len(heights_mm)
    L_effs = []
    for h in heights_mm:
        a = _alpha(float(h), l_bay_edge_mm)
        delta_story = theta_target * float(h)
        delta_sma   = delta_story * math.cos(a)
        L_eff_i     = max(delta_sma / eps_target, 1.0)
        L_effs.append(L_eff_i)
    return L_effs

def _normalize_coeffs(coeffs_raw: List[float], n_story: int) -> List[float]:
    if not coeffs_raw:
        coeffs_raw = [1.0] * n_story
    if len(coeffs_raw) == n_story:
        arr = np.asarray(coeffs_raw, dtype=float)
    else:
        x_old = np.linspace(0, 1, num=len(coeffs_raw))
        x_new = np.linspace(0, 1, num=n_story)
        arr = np.interp(x_new, x_old, np.asarray(coeffs_raw, dtype=float))
    arr = np.clip(arr, 1e-8, None)
    return arr.tolist()

# ==============================
# 1) SelfCentering 阻尼器构建（取消位移限制：不再 MinMax）
# ==============================
def PFSDB(
    SpringID: int,
    NodeI: int,
    NodeJ: int,
    pts: List[float],
    MinMax: Optional[List[float]] = None  # 这里保留形参以兼容旧调用，但会被忽略
):
    """
    pts 至少 5 个：[k1, k2, sigAct, beta, epsSlip]
    若给到 7 个则：[k1, k2, sigAct, beta, epsSlip, epsBear, rBear]
    """
    if len(pts) < 5:
        raise ValueError(f"SelfCentering Error: Expected at least 5 params, got {len(pts)}")

    sc_tag = SpringID

    k1     = float(pts[0])
    k2     = float(pts[1])
    sigAct = float(pts[2])
    beta   = float(pts[3])
    epsSlip = float(pts[4])

    # bearing 参数：默认不启用
    epsBear = float(pts[5]) if len(pts) >= 6 else 0.0
    rBear   = float(pts[6]) if len(pts) >= 7 else k1

    # ---- 安全夹紧：避免报错/发散 ----
    k1 = max(k1, 1e-6)
    k2 = min(max(k2, 0.0), 0.999999 * k1)
    sigAct = max(sigAct, 1e-6)
    beta = float(np.clip(beta, 1e-6, 0.999999))
    epsSlip = max(epsSlip, 0.0)
    epsBear = max(epsBear, 0.0)
    rBear = max(rBear, 1e-6)

    # SelfCentering 标准参数：k1 k2 sigAct beta epsSlip epsBear rBear
    ops.uniaxialMaterial("SelfCentering", sc_tag, k1, k2, sigAct, beta, epsSlip, epsBear, rBear)

    # ===== 取消位移限制：不包 MinMax（外部传 MinMax 也忽略）=====
    ops.element("twoNodeLink", SpringID, NodeI, NodeJ, "-mat", sc_tag, "-dir", 1)

# ==============================
# 2) 经验式：计算 SelfCentering 参数（含 epsSlip）
# ==============================
def get_self_centering_params(T, A_input, L_input):
    """
    返回 pts: [k1, k2, sigAct, beta, epsSlip, epsBear, rBear]
    其中 epsBear=0（先不启用 bearing），rBear=k1
    """
    A = float(A_input)
    L = float(L_input)

    if L < 1.0:
        print(f"Error: 支撑长度 L={L} 异常，重置为 1000mm 防止崩溃")
        L = 1000.0

    # --- 1. 你提供的温度-模量关系（注意单位转换与你原脚本保持一致） ---
    E_slope  = 3.7835
    E_int    = 431.3967

    # 相变温度参数（你原脚本）
    Ms = -72.0104
    Mf = -147.2169
    As = -37.6725
    Af = -16.8237

    scale_factor = 1.5
    CM = 5.7713 * scale_factor
    CA = 6.2409 * scale_factor

    # 当前温度 T 下的相变力（力单位：与你 A(面积) 一致的推导保持一致）
    AM_s = CM * (T - Ms) * A
    AM_f = CM * (T - Mf) * A
    MA_s = CA * (T - As) * A
    MA_f = CA * (T - Af) * A

    # --- 2. 初始刚度 k1 ---
    E_mat = (E_slope * T + E_int) * 100.0  # 与你原脚本一致
    k1 = E_mat * A / L

    # --- 3. 启动力 sigAct ---
    sigAct = max(AM_s, 1e-6)

    # --- 4. 激活后刚度 k2（硬化比，可自行改成温度相关） ---
    k2 = 0.1 * k1

    # --- 5. beta（你当前的“保持耗能宽度近似恒定”的反算方法） ---
    T_ref = 20.0
    AM_s_ref = CM * (T_ref - Ms) * A
    beta_ref = 0.5

    delta_F_constant = AM_s_ref * beta_ref

    if sigAct > 0:
        beta_val = delta_F_constant / sigAct
    else:
        beta_val = 0.0

    beta_val = float(np.clip(beta_val, 0.01, 0.99))

    # --- 6. epsSlip：用 6% 应变对应位移（你要求的从 6% 开始） ---
    slip_strain = 0.06
    epsSlip = slip_strain * L

    # --- 7. bearing：先关闭 ---
    epsBear = 0.0
    rBear = k1

    return [k1, k2, sigAct, beta_val, epsSlip, epsBear, rBear]

# ==============================
# 3) 外部接口：返回每层 SelfCentering 参数（仅返回 pts_list，取消 minmax）
# ==============================
def PFSDB_pts(T: float = 20.0) -> Tuple[List[float], ...]:
    cfg = _load_iter_cfg()

    # 默认值
    heights_mm    = [5500] + [4300] * 7
    n_story       = 8
    A_base        = 35000.0
    coeffs        = [1.0] * n_story
    l_bay_edge_mm = 9150.0
    L1            = 0.0
    L2            = 0.0

    if cfg:
        heights_mm    = cfg.get("heights_mm", heights_mm)
        n_story       = len(heights_mm)
        l_bay_edge_mm = float(cfg.get("L_bay_edge_mm", l_bay_edge_mm))
        L1            = float(cfg.get("L1", 0.0))
        L2            = float(cfg.get("L2", 0.0))
        T             = float(cfg.get("T", T))

        if "A_base_mm2" in cfg:
            A_base = float(cfg["A_base_mm2"])
        else:
            A_guess = float(cfg.get("A", 35000.0))
            A_base = A_guess if A_guess > 1000 else 35000.0

        coeffs = _normalize_coeffs(cfg.get("coeffs", []), n_story)

    # 计算 L1/L2（几何斜长）
    if L1 < 1.0:
        L1 = math.sqrt(heights_mm[0] ** 2 + (l_bay_edge_mm / 2) ** 2)
    if L2 < 1.0 and len(heights_mm) > 1:
        L2 = math.sqrt(heights_mm[1] ** 2 + (l_bay_edge_mm / 2) ** 2)
    elif L2 < 1.0:
        L2 = L1

    pts_list: List[List[float]] = []
    for i in range(n_story):
        A_i = A_base * float(coeffs[i])
        L_i = L1 if i == 0 else L2
        pts_i = get_self_centering_params(T, A_i, L_i)
        pts_list.append(pts_i)

    # 只返回每层 pts（取消 minmax1/minmax2）
    return tuple(pts_list)

# ==============================
# 4) 测试
# ==============================
if __name__ == "__main__":
    print("计算参数 T=20C...")
    pts_only = PFSDB_pts(T=20.0)

    print(f"层数: {len(pts_only)}")

    if len(pts_only) > 6:
        print("\n第7层(Index 6) SelfCentering 参数(7个):")
        print(f"  -> 初始刚度 k1     : {pts_only[6][0]:.2f} N/mm")
        print(f"  -> 激活后刚度 k2   : {pts_only[6][1]:.2f} N/mm")
        print(f"  -> 启动力 sigAct   : {pts_only[6][2]:.2f} N")
        print(f"  -> 耗能系数 beta   : {pts_only[6][3]:.4f}")
        print(f"  -> epsSlip(6%应变) : {pts_only[6][4]:.2f} mm")
        print(f"  -> epsBear         : {pts_only[6][5]:.2f} mm")
        print(f"  -> rBear           : {pts_only[6][6]:.2f} N/mm")
