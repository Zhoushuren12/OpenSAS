import openseespy.opensees as ops
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import json
import math
from pathlib import Path

# ==============================
# 1) twoNodeLink + SelfCentering 材料封装 (SMADB)
# ==============================
def SMADB(
        SpringID: int, NodeI: int, NodeJ: int,
        pts: List[float], MinMax: Optional[List[float]] = None
):
    """
    使用带有 Bearing 硬化功能的 SelfCentering 模型。
    pts 严格按照 OpenSees 顺序: [k1, k2, sigAct, beta, slip_disp, epsBear(Dm), rBear(Km)]
    """
    if len(pts) < 7:
        raise ValueError(f"SMABD Error: Expected at least 7 params, got {len(pts)}")

    # 定义 SelfCentering 材料 (传入 7 个参数)
    ops.uniaxialMaterial("SelfCentering", SpringID, *pts[:7])

    # 如果有 MinMax 限制，则包裹一层
    if MinMax is not None:
        minmax_material_id = SpringID * 100  # 确保 ID 唯一且不冲突
        ops.uniaxialMaterial("MinMax", minmax_material_id, SpringID,
                             "-min", MinMax[0], "-max", MinMax[1])
        ops.element("twoNodeLink", SpringID, NodeI, NodeJ,
                    "-mat", minmax_material_id, "-dir", 1)
    else:
        ops.element("twoNodeLink", SpringID, NodeI, NodeJ,
                    "-mat", SpringID, "-dir", 1)


# ==============================
# 2) 材料参数物理计算 (包含二阶硬化机制)
# ==============================
def get_self_centering_params(T, A_input, L_input):
    A = float(A_input)
    L = float(L_input)
    
    if L < 1.0: 
        print(f"Error: 支撑长度 L={L} 异常，重置为 1000mm 防止崩溃")
        L = 1000.0

    # --- 1. 核心物理参数 ---
    E_slope  = 3.7835
    E_int    = 431.3967

    Ms       = -72.0104
    Mf       = -147.2169
    CA       = 6.1659
    As       = -37.6725
    Af       = -16.8237

    scale_factor = 1.5
    CM = 5.7713 * scale_factor
    CA_scaled = 6.2409 * scale_factor   

    # 当前温度 T 下的相变力
    AM_s = CM * (T - Ms) * A 
    AM_f = CM * (T - Mf) * A 
    MA_s = CA_scaled * (T - As) * A 
    MA_f = CA_scaled * (T - Af) * A 

    # --- 2. 初始刚度 k1 ---
    E_mat = (E_slope * T + E_int) * 100.0
    k1 = E_mat * A / L 

    # --- 3. 启动力 sigAct (奥氏体 -> 马氏体 起始) ---
    sigAct = AM_s
    
    # --- 4. 屈服后刚度 k2 (相变平台段) ---
    k2 = 0.1 * k1 

    # --- 5. 耗能参数 beta ---
    T_ref = 20.0
    AM_s_ref = CM * (T_ref - Ms) * A 
    beta_ref = 0.5
    
    # 锁定绝对的滞回环高度差
    delta_F_constant = AM_s_ref * beta_ref
    
    if sigAct > 0:
        beta = delta_F_constant / sigAct
    else:
        beta = 0.0
        
    beta = np.clip(beta, 0.01, 0.99)

    # --- 6. 滑移间隙 ---
    slip_disp = 0.0

    # --- 7. 硬化 (马氏体强化段) ---
    Km = 0.5          # 硬化刚度比 (rBear)
    Dm = 0.06 * L     # 触发硬化的位移变形量 (epsBear)

    # 返回参数列表 (注意: Dm 必须在 Km 前面，以符合 OpenSees 语法)
    return [k1, k2, sigAct, beta, slip_disp, Dm, Km]


# ==============================
# 3) 读取迭代配置
# ==============================
_CFG_CACHE = None
_CFG_MTIME = None

def _load_iter_cfg() -> Optional[Dict[str, Any]]:
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
            content = cfg_path.read_text(encoding="utf-8")
            _CFG_CACHE = json.loads(content)
            _CFG_MTIME = stat.st_mtime
        except json.JSONDecodeError:
            print("Warning: json 文件格式错误，返回 None")
            return None
    return _CFG_CACHE

def _normalize_coeffs(coeffs_raw: List[float], n_story: int) -> List[float]:
    if not coeffs_raw:
        return [1.0] * n_story
    
    arr = np.array(coeffs_raw, dtype=float)
    if len(arr) != n_story:
        x_old = np.linspace(0, 1, num=len(arr))
        x_new = np.linspace(0, 1, num=n_story)
        arr = np.interp(x_new, x_old, arr)
        
    arr = np.clip(arr, 1e-6, None)
    return arr.tolist()


# ==============================
# 4) 核心：生成每层 SMADB 参数
# ==============================
def SMADB_pts(T: float = 20.0) -> Tuple:
    cfg = _load_iter_cfg()

    if cfg is None:
        heights_mm = [5500] + [4300]*7 
        A_base = 35000.0
        coeffs = [1.0] * len(heights_mm)
        L_bay_edge_mm = 9150.0
        L1, L2 = 0.0, 0.0 
        minmax1, minmax2 = None, None
    else:
        heights_mm = cfg.get("heights_mm", [5500] + [4300]*7)
        if "A_base_mm2" in cfg:
            A_base = float(cfg["A_base_mm2"])
        else:
            A_guess = float(cfg.get("A", 35000.0))
            A_base = A_guess if A_guess > 1000 else 35000.0
            
        n_story = len(heights_mm)
        coeffs = _normalize_coeffs(cfg.get("coeffs", []), n_story)
        L_bay_edge_mm = float(cfg.get("L_bay_edge_mm", 9150.0))
        L1 = float(cfg.get("L1", 0.0))
        L2 = float(cfg.get("L2", 0.0))
        minmax1 = cfg.get("minmax1", None)
        minmax2 = cfg.get("minmax2", None)

    if L1 < 1.0:
        L1 = math.sqrt(heights_mm[0]**2 + (L_bay_edge_mm/2)**2)
    
    if L2 < 1.0 and len(heights_mm) > 1:
        L2 = math.sqrt(heights_mm[1]**2 + (L_bay_edge_mm/2)**2)
    elif L2 < 1.0:
        L2 = L1 

    pts_list: List[List[float]] = []
    n_story = len(heights_mm)

    for i in range(n_story):
        A_i = A_base * float(coeffs[i])
        L_i = L1 if i == 0 else L2

        # 调用核心函数，返回 7 个参数
        pts_i = get_self_centering_params(T, A_i, L_i)
        pts_list.append(pts_i)

    return tuple(pts_list) + (minmax1, minmax2)


# ==============================
# 5) 测试
# ==============================
if __name__ == "__main__":
    print("--- Calculating SMADB Parameters (T=20C) ---")
    out = SMADB_pts(T=20.0)
    
    n_items = len(out)
    pts_all = out[:n_items-2] 
    minmax1 = out[-2]
    minmax2 = out[-1]

    print(f"Total layers: {len(pts_all)}")
    if len(pts_all) > 0:
        print(f"\nLayer 1 SMADB Params (k1, k2, sigAct, beta, slip, Dm, Km): {pts_all[0]}")
        print(f"  -> Initial Stiffness k1    : {pts_all[0][0]:.2f} N/mm")
        print(f"  -> Yield Stiffness k2      : {pts_all[0][1]:.2f} N/mm (5% of k1)")
        print(f"  -> Yield Force sigAct      : {pts_all[0][2]:.2f} N")
        print(f"  -> Energy beta             : {pts_all[0][3]:.4f}")
        print(f"  -> Slip Displacement       : {pts_all[0][4]:.2f} mm")
        print(f"  -> Hardening Trigger Dm    : {pts_all[0][5]:.2f} mm (6% Strain)")
        print(f"  -> Hardening Stiffness Km  : {pts_all[0][6]:.2f} (Ratio to k1)")
        
    if len(pts_all) > 6:
        print(f"\nLayer 7 SMADB Params: {pts_all[6]}")
