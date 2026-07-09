from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# =========================
# 全局绘图设置
# =========================
plt.rc('font', family='Times New Roman')
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 路径设置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / 'Output_data'
SPEC_DIR = PROJECT_ROOT / 'Spectrum'
OUT_DIR = PROJECT_ROOT / 'Paint'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_y(x: list | np.ndarray, y: list | np.ndarray, x0: float, error: bool = True) -> float | None:
    """获得竖线 x=x0 与给定曲线的交点纵坐标（线性插值）"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size != y.size:
        raise ValueError("x 和 y 长度不一致。")
    if x.size < 2:
        raise ValueError("用于插值的数据点少于 2 个。")

    if x0 < np.min(x):
        if error:
            raise ValueError(f'【Error】x0 < min(x) ({x0} < {np.min(x)})')
        return None

    if x0 > np.max(x):
        if error:
            raise ValueError(f'【Error】x0 > max(x) ({x0} > {np.max(x)})')
        return None

    for i in range(len(x) - 1):
        if x[i] == x0:
            return float(y[i])
        elif x[i] < x0 <= x[i + 1]:
            k = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            return float(k * (x0 - x[i]) + y[i])

    raise ValueError('【Error】未找到交点。请检查 x 是否单调递增。')


# =========================================
# 1. 50 年超越概率 -> 年超越频率 MAF
#    λ = -ln(1 - P50) / 50
# =========================================
def prob50_to_maf(P_50, life=50.0):
    P_50 = np.asarray(P_50, dtype=float)
    if np.any((P_50 <= 0) | (P_50 >= 1)):
        raise ValueError("P_50 必须在 (0,1) 范围内。")
    return -np.log(1.0 - P_50) / life


# =========================================
# 2. Bradley hazard 模型
#    λ(IM) = η1 * exp[ η2 / ln(IM / η3) ]
# =========================================
def bradley_lambda(IM, eta1, eta2, eta3):
    IM = np.asarray(IM, dtype=float)

    if np.any(IM <= 0):
        raise ValueError("IM 必须为正数。")

    # 为了便于 curve_fit 搜索，这里不直接 raise，而是返回大值惩罚
    if eta1 <= 0 or eta3 <= 0:
        return np.full_like(IM, 1e20, dtype=float)

    log_term = np.log(IM / eta3)

    # 避免 ln(IM/eta3)=0 的奇异点
    if np.any(np.isclose(log_term, 0.0, atol=1e-12)):
        return np.full_like(IM, 1e20, dtype=float)

    value = eta1 * np.exp(eta2 / log_term)

    if np.any(~np.isfinite(value)):
        return np.full_like(IM, 1e20, dtype=float)

    return value


# =========================================
# 3. 拟合 Bradley hazard 参数
# =========================================
def fit_bradley_hazard(IM_vals, maf_vals):
    """
    拟合模型:
        λ(IM) = η1 * exp[ η2 / ln(IM / η3) ]
    """
    IM_vals = np.asarray(IM_vals, dtype=float)
    maf_vals = np.asarray(maf_vals, dtype=float)

    if len(IM_vals) < 3:
        raise ValueError("至少需要 3 个点拟合 Bradley 模型。")
    if np.any(IM_vals <= 0) or np.any(maf_vals <= 0):
        raise ValueError("IM 和 λ 必须都为正数。")

    # 初值
    eta1_0 = float(np.median(maf_vals))
    eta2_0 = -2.0
    eta3_0 = float(np.max(IM_vals) * 1.2)

    p0 = [eta1_0, eta2_0, eta3_0]

    # 约束
    lower_bounds = [1e-12, -1e3, 1e-6]
    upper_bounds = [1e2,  1e3,  1e2]

    popt, _ = curve_fit(
        bradley_lambda,
        IM_vals,
        maf_vals,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=50000,
    )

    eta1, eta2, eta3 = popt
    return float(eta1), float(eta2), float(eta3)


# =========================================
# 4. 保存拟合摘要
# =========================================
def export_fit_summary(IM_vals, maf_vals, eta1, eta2, eta3, out_txt_path, out_csv_path):
    IM_vals = np.asarray(IM_vals, dtype=float)
    maf_vals = np.asarray(maf_vals, dtype=float)
    maf_fit = bradley_lambda(IM_vals, eta1, eta2, eta3)

    out_txt_path = Path(out_txt_path)
    out_csv_path = Path(out_csv_path)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_txt_path, 'w', encoding='utf-8') as f:
        f.write("Bradley hazard model\n")
        f.write("lambda(IM) = eta1 * exp[eta2 / ln(IM / eta3)]\n\n")
        f.write(f"eta1 = {eta1:.10e}\n")
        f.write(f"eta2 = {eta2:.10e}\n")
        f.write(f"eta3 = {eta3:.10e}\n\n")
        f.write("Back-check at input points:\n")
        for i, (im, maf_true, maf_pred) in enumerate(zip(IM_vals, maf_vals, maf_fit), start=1):
            rel_err = abs(maf_pred - maf_true) / maf_true
            f.write(
                f"Point {i}: IM = {im:.6f}, "
                f"true = {maf_true:.10e}, fitted = {maf_pred:.10e}, "
                f"rel.err = {rel_err:.6e}\n"
            )

    data = np.column_stack([IM_vals, maf_vals, maf_fit, np.abs(maf_fit - maf_vals) / maf_vals])
    np.savetxt(
        out_csv_path,
        data,
        fmt="%.6e",
        delimiter=",",
        header="IM,MAF_true,MAF_fitted,relative_error",
        comments=""
    )


# =========================================
# 5. 导出连续 hazard 曲线数据
# =========================================
def export_hazard_curve(IM_grid, lambda_grid, save_curve_path):
    save_curve_path = Path(save_curve_path)
    save_curve_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        save_curve_path,
        np.column_stack([IM_grid, lambda_grid]),
        fmt="%.6e",
        header="IM lambda(IM)",
        comments=""
    )


# =========================================
# 6. 绘制 hazard 曲线
# =========================================
def plot_hazard_curve(
    IM_vals,
    maf_vals,
    eta1,
    eta2,
    eta3,
    IM_min=None,
    IM_max=None,
    n_points=500,
    point_labels=None,
    point_colors=None,
    fig_png_path=None,
    fig_pdf_path=None,
):
    IM_vals = np.asarray(IM_vals, dtype=float)
    maf_vals = np.asarray(maf_vals, dtype=float)

    if IM_min is None:
        IM_min = max(1e-4, IM_vals.min() * 0.7)
    if IM_max is None:
        IM_max = IM_vals.max() * 1.3

    IM_grid = np.linspace(IM_min, IM_max, n_points)
    lambda_grid = bradley_lambda(IM_grid, eta1, eta2, eta3)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    # 原始离散点
    if point_labels is None:
        ax.semilogy(IM_vals, maf_vals, 'o', color='k', label='Input points')
    else:
        colors = point_colors or ['k'] * len(IM_vals)
        for x, y, label, color in zip(IM_vals, maf_vals, point_labels, colors):
            ax.semilogy(
                [x], [y], 'o',
                markerfacecolor='none',
                markeredgecolor=color,
                markeredgewidth=2.0,
                markersize=9,
                label=label
            )

    # 拟合曲线
    ax.semilogy(IM_grid, lambda_grid, '-', color='k', linewidth=1.8, label='Fitted hazard curve')

    ax.tick_params(axis='both', direction='in', which='both', labelsize=16, top=True, right=True)
    ax.set_xlabel(r'$Sa(T_1)$ (g)', fontsize=18, labelpad=10)
    ax.set_ylabel('MAF', fontsize=18, labelpad=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.45)
    ax.legend(fontsize=12, frameon=False)

    ax.set_xlim(0, IM_max)
    y_min = min(np.min(maf_vals), np.min(lambda_grid))
    y_max = max(np.max(maf_vals), np.max(lambda_grid))
    ax.set_ylim(max(1e-6, y_min * 0.5), y_max * 2.0)

    plt.tight_layout()

    if fig_png_path is not None:
        fig_png_path = Path(fig_png_path)
        fig_png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_png_path, dpi=600, bbox_inches='tight')

    if fig_pdf_path is not None:
        fig_pdf_path = Path(fig_pdf_path)
        fig_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_pdf_path, bbox_inches='tight')

    plt.close(fig)

    return IM_grid, lambda_grid


# =========================================
# 7. 主程序
# =========================================
if __name__ == "__main__":
    # ---------- ① 读取结构一阶周期 ----------
    T0 = np.loadtxt(BASE_DIR / 'MC8_SMABF' / 'MC8_PO_out' / '周期(s).out')[0]

    # ---------- ② 读取 FOE / DBE / MCE 反应谱 ----------
    data_FOE = np.loadtxt(SPEC_DIR / 'FOE Level Spectrum.txt')
    data_DBE = np.loadtxt(SPEC_DIR / 'DBE Level Spectrum.txt')
    data_MCE = np.loadtxt(SPEC_DIR / 'MCE Level Spectrum.txt')

    T_FOE, Sa_FOE_arr = data_FOE[:, 0], data_FOE[:, 1]
    T_DBE, Sa_DBE_arr = data_DBE[:, 0], data_DBE[:, 1]
    T_MCE, Sa_MCE_arr = data_MCE[:, 0], data_MCE[:, 1]

    # ---------- ③ 在 T1 处插值得到 Sa(T1) ----------
    Sa_FOE = get_y(T_FOE, Sa_FOE_arr, T0)
    Sa_DBE = get_y(T_DBE, Sa_DBE_arr, T0)
    Sa_MCE = get_y(T_MCE, Sa_MCE_arr, T0)

    IM_vals = np.array([Sa_FOE, Sa_DBE, Sa_MCE], dtype=float)

    # ---------- ④ 输入 50 年超越概率 ----------
    P_FOE_50 = 0.63
    P_DBE_50 = 0.10
    P_MCE_50 = 0.02

    P_50 = np.array([P_FOE_50, P_DBE_50, P_MCE_50], dtype=float)

    # ---------- ⑤ 转换为年超越频率 ----------
    MAF_vals = prob50_to_maf(P_50, life=50.0)

    # ---------- ⑥ 拟合 Bradley 参数 ----------
    eta1, eta2, eta3 = fit_bradley_hazard(IM_vals, MAF_vals)

    print("Bradley 拟合系数：")
    print(f"  eta1 = {eta1:.10e}")
    print(f"  eta2 = {eta2:.10e}")
    print(f"  eta3 = {eta3:.10e}")

    print("\n对应 hazard 曲线：")
    print("  lambda(IM) = eta1 * exp[eta2 / ln(IM / eta3)]")
    print(f"  lambda(IM) = {eta1:.6e} * exp({eta2:.6e} / ln(IM / {eta3:.6e}))")

    # ---------- ⑦ 三点回代检查 ----------
    maf_fit = bradley_lambda(IM_vals, eta1, eta2, eta3)
    print("\n三点回代检查：")
    for im, maf_true, maf_pred in zip(IM_vals, MAF_vals, maf_fit):
        rel_err = abs(maf_pred - maf_true) / maf_true
        print(
            f"IM = {im:.6f}, true = {maf_true:.6e}, "
            f"fitted = {maf_pred:.6e}, rel.err = {rel_err:.6e}"
        )

    # ---------- ⑧ 输出路径 ----------
    fig_png_path = OUT_DIR / "hazard_curve_bradley.png"
    fig_pdf_path = OUT_DIR / "hazard_curve_bradley.pdf"
    curve_out_path = OUT_DIR / "hazard_fitted_curve_bradley.txt"
    param_txt_path = OUT_DIR / "hazard_fit_params_bradley.txt"
    param_csv_path = OUT_DIR / "hazard_fit_points_bradley.csv"

    # ---------- ⑨ 画图 ----------
    point_labels = [
        'FOE (63% in 50 years)',
        'DBE (10% in 50 years)',
        'MCE (2% in 50 years)'
    ]
    point_colors = ['tab:green', 'tab:blue', 'tab:orange']

    IM_grid, lambda_grid = plot_hazard_curve(
        IM_vals, MAF_vals,
        eta1, eta2, eta3,
        IM_min=max(0.01, np.min(IM_vals) * 0.7),
        IM_max=np.max(IM_vals) * 1.3,
        point_labels=point_labels,
        point_colors=point_colors,
        fig_png_path=fig_png_path,
        fig_pdf_path=fig_pdf_path,
    )

    # ---------- ⑩ 导出拟合摘要 ----------
    export_fit_summary(
        IM_vals, MAF_vals,
        eta1, eta2, eta3,
        out_txt_path=param_txt_path,
        out_csv_path=param_csv_path
    )

    # ---------- ⑪ 导出连续曲线 ----------
    export_hazard_curve(IM_grid, lambda_grid, curve_out_path)

    print(f"\n已保存图像: {fig_png_path}")
    print(f"已保存图像: {fig_pdf_path}")
    print(f"已保存连续 hazard 曲线: {curve_out_path}")
    print(f"已保存拟合参数摘要: {param_txt_path}")
    print(f"已保存拟合点检查表: {param_csv_path}")