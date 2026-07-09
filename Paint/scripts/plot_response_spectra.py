"""地震动反应谱与设计谱对比绘图工具。

用途：叠加全部地震动反应谱、平均谱和规范设计谱。
做法：读取 ``GMs`` 中的加速度记录及时间步长，参照 ``GM_spectrum.py`` 的 ``Spectrum()``
精确递推法计算 5% 阻尼谱，并读取 ``Spectrum`` 目录下的设计谱。
使用：修改顶部“用户编辑区”中的路径、阻尼比和周期范围后运行本文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_common import PROJECT_ROOT, configure_matplotlib, save_figure, style_axes


# ========================= 用户编辑区 =========================
GMS_DIR = PROJECT_ROOT / "GMs"
GM_INFO_FILE = GMS_DIR / "GM_info.json"
DESIGN_SPECTRUM_FILE = PROJECT_ROOT / "Spectrum" / "MCE.txt"
OUTPUT_FILE = PROJECT_ROOT / "Paint" / "Figures" / "response_spectra.png"

DAMPING_RATIO = 0.05
MAX_PERIOD = 4.0
PERIOD_STEP = 0.02
SHOW_FIGURE = True
# ======================= 用户编辑区结束 =======================


def natural_key(path: Path) -> tuple[int, int | str]:
    """Sort numeric record names numerically and other names alphabetically."""
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.stem.lower()


def load_time_steps(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig") as file:
        raw = json.load(file)

    time_steps = {str(name): float(dt) for name, dt in raw.items()}
    invalid = {name: dt for name, dt in time_steps.items() if dt <= 0.0}
    if invalid:
        raise ValueError(f"GM_info.json 中存在非正时间步长: {invalid}")
    return time_steps


def load_ground_motion(path: Path) -> np.ndarray:
    acceleration = np.asarray(np.loadtxt(path, dtype=float), dtype=float).reshape(-1)
    if acceleration.size < 2:
        raise ValueError(f"地震动文件至少需要两个数据点: {path}")
    if not np.all(np.isfinite(acceleration)):
        raise ValueError(f"地震动文件包含非有限数值: {path}")
    return acceleration


def load_design_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column period-Sa file, with or without one text header row."""
    data = np.genfromtxt(path, dtype=float, comments="#", invalid_raise=False)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"设计谱文件应至少包含两列（周期、Sa）: {path}")

    period = data[:, 0]
    sa = data[:, 1]
    valid = np.isfinite(period) & np.isfinite(sa)
    period, sa = period[valid], sa[valid]
    if period.size < 2:
        raise ValueError(f"设计谱文件中的有效数据不足: {path}")

    order = np.argsort(period)
    return period[order], sa[order]


def Spectrum(
    ag: np.ndarray,
    dt: float,
    T: np.ndarray,
    zeta: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate RSA, RSV and RSD using the reference exact recurrence.

    The equations follow ``Acceleration_spectrum/tools/GM_spectrum.py::Spectrum``.
    Only the current displacement and velocity vectors are retained; this gives
    the same peak responses without allocating full ``period × time`` matrices.
    """
    ag = np.asarray(ag, dtype=float).reshape(-1)
    T = np.asarray(T, dtype=float).reshape(-1)
    if ag.size < 2 or not np.all(np.isfinite(ag)):
        raise ValueError("地震动数组至少需要两个有限数值。")
    if dt <= 0.0:
        raise ValueError("时间步长 dt 必须大于 0。")
    if T.size == 0 or not np.all(np.isfinite(T)):
        raise ValueError("周期数组 T 必须包含有限数值。")
    if not 0.0 <= zeta < 1.0:
        raise ValueError("阻尼比必须满足 0 <= zeta < 1。")

    # 与参考函数一致：当首个周期为 0 时，以 PGA 作为 RSA(0)。
    mark = bool(np.isclose(T[0], 0.0))
    positive_periods = T[1:] if mark else T
    if np.any(positive_periods <= 0.0):
        raise ValueError("除首个可选零周期外，其余周期必须大于 0。")
    if positive_periods.size == 0:
        return (
            np.array([np.max(np.abs(ag))]),
            np.array([0.0]),
            np.array([0.0]),
        )

    w = 2.0 * np.pi / positive_periods
    wd = w * np.sqrt(1.0 - zeta**2)
    B1 = np.exp(-zeta * w * dt) * np.cos(wd * dt)
    B2 = np.exp(-zeta * w * dt) * np.sin(wd * dt)
    w_2 = 1.0 / w**2
    w_3 = 1.0 / w**3

    u = np.zeros_like(positive_periods)
    v = np.zeros_like(positive_periods)
    RSA = np.zeros_like(positive_periods)
    RSV = np.zeros_like(positive_periods)
    RSD = np.zeros_like(positive_periods)

    for index in range(ag.size - 1):
        p_i = -ag[index]
        alpha_i = (-ag[index + 1] + ag[index]) / dt

        A0 = p_i * w_2 - 2.0 * zeta * alpha_i * w_3
        A1 = alpha_i * w_2
        A2 = u - A0
        A3 = (v + zeta * w * A2 - A1) / wd

        u = A0 + A1 * dt + A2 * B1 + A3 * B2
        v = A1 + (wd * A3 - zeta * w * A2) * B1 - (wd * A2 + zeta * w * A3) * B2
        a = -2.0 * zeta * w * v - w * w * u

        RSA = np.maximum(RSA, np.abs(a))
        RSV = np.maximum(RSV, np.abs(v))
        RSD = np.maximum(RSD, np.abs(u))

    if mark:
        RSA = np.insert(RSA, 0, np.max(np.abs(ag)))
        RSV = np.insert(RSV, 0, 0.0)
        RSD = np.insert(RSD, 0, 0.0)
    return RSA, RSV, RSD


def acceleration_response_spectrum(
    acceleration: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping_ratio: float = 0.05,
) -> np.ndarray:
    """Backward-compatible RSA-only wrapper around :func:`Spectrum`."""
    RSA, _, _ = Spectrum(acceleration, dt, periods, damping_ratio)
    return RSA


def calculate_ground_motion_spectra(
    gm_dir: Path,
    gm_info_file: Path,
    periods: np.ndarray,
    damping_ratio: float,
) -> tuple[list[Path], np.ndarray]:
    time_steps = load_time_steps(gm_info_file)
    gm_files = sorted(gm_dir.glob("*.txt"), key=natural_key)
    if not gm_files:
        raise FileNotFoundError(f"未找到地震动文件: {gm_dir / '*.txt'}")

    missing_dt = [path.name for path in gm_files if path.stem not in time_steps]
    if missing_dt:
        raise KeyError(f"GM_info.json 缺少以下记录的时间步长: {missing_dt}")

    spectra = []
    for path in gm_files:
        acceleration = load_ground_motion(path)
        RSA, _, _ = Spectrum(
            acceleration,
            time_steps[path.stem],
            periods,
            damping_ratio,
        )
        spectra.append(RSA)
    return gm_files, np.vstack(spectra)


def plot_response_spectra() -> Path:
    configure_matplotlib()
    periods = np.arange(0.0, MAX_PERIOD + PERIOD_STEP / 2.0, PERIOD_STEP)
    gm_files, spectra = calculate_ground_motion_spectra(
        GMS_DIR,
        GM_INFO_FILE,
        periods,
        DAMPING_RATIO,
    )
    design_period, design_sa = load_design_spectrum(DESIGN_SPECTRUM_FILE)
    mean_spectrum = np.mean(spectra, axis=0)

    fig, ax = plt.subplots(figsize=(8,6))
    for index, spectrum in enumerate(spectra):
        ax.plot(
            periods,
            spectrum,
            color="0.72",
            alpha=1,
            linewidth=0.8,
            label="Ground motion spectra" if index == 0 else None,
            zorder=1,
        )
    ax.plot(
        design_period,
        design_sa,
        color="black",
        linestyle="-",
        linewidth=2.0,
        label="Design spectrum",
        zorder=3,
    )
    ax.plot(
        periods,
        mean_spectrum,
        color="#1f77b4",
        linewidth=2.4,
        label="Mean spectrum",
        zorder=4,
    )

    style_axes(ax, xlabel="Period (s)", ylabel=r"$S_a$ (g)")
    ax.set_xlim(0.0, MAX_PERIOD)
    ax.set_ylim(bottom=0.0)
    ax.minorticks_on()
    ax.grid(which="major", color="0.86", linewidth=0.8)
    ax.grid(which="minor", color="0.92", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right", frameon=False, fontsize=16)

    output_path = save_figure(fig, OUTPUT_FILE)
    print(
        f"已读取 {len(gm_files)} 条地震动，阻尼比 {DAMPING_RATIO:.0%}；"
        f"图形已保存至: {output_path}"
    )
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
    return output_path


if __name__ == "__main__":
    plot_response_spectra()
