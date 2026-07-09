from pickle import NONE
import warnings
from pathlib import Path
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rc("font", family="Times New Roman")
plt.rc("mathtext", fontset="stix")


MODEL_ALIASES = {
    "SAMBF": "SMABF",
}


def _infer_dm_name(file_paths: List[Union[str, Path]], xlabel: Optional[str] = None) -> Optional[str]:
    for fp in file_paths:
        stem = Path(fp).stem.upper()
        for dm_name in ("RIDR", "IDR", "PFA"):
            if stem == dm_name or stem.endswith(f"_{dm_name}"):
                return dm_name

    if xlabel:
        normalized = xlabel.upper().replace(" ", "")
        for dm_name in ("RIDR", "IDR", "PFA"):
            if normalized in {dm_name, f"{dm_name}(%)"}:
                return dm_name

    return None


def _dm_axis_scale(dm_name: Optional[str]) -> float:
    return 100.0 if dm_name in {"IDR", "RIDR"} else 1.0


def _format_xlabel(dm_name: Optional[str], xlabel: Optional[str], x_scale: float) -> str:
    if not xlabel:
        if dm_name == "PFA":
            return "PFA(g)"
        return f"{dm_name} (%)" if x_scale != 1.0 and dm_name else (dm_name or "DM")

    normalized = xlabel.upper().replace(" ", "")
    if dm_name and normalized in {"IDR", "RIDR", "PFA", "IDR(%)", "RIDR(%)", "PFA(%)"}:
        if dm_name == "PFA":
            return "PFA(g)"
        return f"{dm_name} (%)" if x_scale != 1.0 else dm_name

    if x_scale != 1.0 and "%" not in xlabel:
        return f"{xlabel} (%)"
    return xlabel


def _scale_axis_limits(limits: Optional[Tuple[float, float]], scale: float) -> Optional[Tuple[float, float]]:
    if limits is None or scale == 1.0:
        return limits
    return tuple(None if v is None else float(v) * scale for v in limits)


def _is_index_like(col: np.ndarray) -> bool:
    col = col[np.isfinite(col)]
    if col.size < 2:
        return False
    if not np.all(np.isclose(col, np.round(col))):
        return False
    ints = col.astype(int)
    if np.any(np.diff(ints) < 0):
        return False
    if ints[0] not in (0, 1):
        return False
    diffs = np.diff(ints)
    return np.all((diffs == 0) | (diffs == 1))


def read_ida_pairs(
    file_path: Union[str, Path],
    sheet: Union[str, int] = 0,
    collapse_limit: Optional[float] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Read [DM1, IM1, DM2, IM2, ...] sheet -> list of (DM, IM) arrays.
    Clean: drop NaN/neg, sort by DM, apply collapse_limit, prepend (0,0).
    """
    file_path = Path(file_path)
    df = pd.read_excel(file_path, sheet_name=sheet, header=None).apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    arr = df.to_numpy()

    if arr.shape[1] % 2 != 0 and arr.shape[1] >= 3 and _is_index_like(arr[:, 0]):
        warnings.warn(f"Index-like first column detected; dropping it. sheet={sheet}", RuntimeWarning)
        arr = arr[:, 1:]

    if arr.shape[1] % 2 != 0:
        raise ValueError(f"Expect even columns (DM/IM pairs), got {arr.shape[1]}.")

    pairs = []
    for k in range(arr.shape[1] // 2):
        dm = arr[:, 2 * k]
        im = arr[:, 2 * k + 1]
        ok = np.isfinite(dm) & np.isfinite(im) & (dm >= 0) & (im >= 0)
        dm, im = dm[ok], im[ok]
        if dm.size < 2:
            continue

        idx = np.argsort(dm)
        dm, im = dm[idx], im[idx]

        if collapse_limit is not None:
            hit = dm >= float(collapse_limit)
            if np.any(hit):
                j = int(np.argmax(hit))
                dm, im = dm[: j + 1], im[: j + 1]

        pairs.append((np.r_[0.0, dm].astype(float), np.r_[0.0, im].astype(float)))

    if len(pairs) < 2:
        raise ValueError("Need >=2 valid curves.")
    return pairs


def quantiles_on_grid(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    *,
    density: int = 800,
    x_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized quantiles (16/50/84) on a common DM grid."""
    x1 = min(dm.min() for dm, _ in pairs)
    x2 = max(dm.max() for dm, _ in pairs)
    if x_max is not None:
        x2 = min(x2, float(x_max))
    if x2 < x1:
        raise ValueError(f"x_max={x_max} 小于数据最小值 {x1}")
    x = np.linspace(x1, x2, density)

    Y = np.full((len(pairs), x.size), np.nan, dtype=float)
    for i, (dm, im) in enumerate(pairs):
        y = np.interp(x, dm, im)
        mask = (x >= dm.min()) & (x <= dm.max())
        Y[i, mask] = y[mask]

    q16 = np.nanpercentile(Y, 16, axis=0)
    q50 = np.nanpercentile(Y, 50, axis=0)
    q84 = np.nanpercentile(Y, 84, axis=0)
    return x, q16, q50, q84


def _read_t1_from_model_dir(model_dir: Path) -> float:
    """Read T1 from model output folder."""
    candidates = [
        model_dir / "MC8_PO_out" / "周期(s).out",
        model_dir / "MC8_IDA_data_out" / "周期(s).out",
    ]
    for p in candidates:
        if p.exists():
            arr = np.asarray(np.loadtxt(p), dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0 and arr[0] > 0:
                return float(arr[0])
            raise ValueError(f"周期文件内容无效: {p}")
    raise FileNotFoundError(f"未找到周期文件: {candidates[0]} 或 {candidates[1]}")


def _infer_model_dir_from_ida_file(file_path: Union[str, Path]) -> Path:
    """Infer model root like Output_data/MC8_xxx from an IDA excel path."""
    fp = Path(file_path).resolve()

    if fp.parent.name in {"MC8_IDA_data_frag", "MC8_IDA_data_out", "MC8_IDA_data"}:
        return fp.parent.parent

    for p in fp.parents:
        if p.name.startswith("MC8_"):
            return p
    raise ValueError(f"无法从路径推断模型目录: {fp}")


def _interp_sa_at_t1(spec_path: Union[str, Path], t1: float) -> float:
    """Interpolate Sa(T1) from spectrum file [T, Sa]."""
    data = np.asarray(np.loadtxt(spec_path), dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"反应谱文件格式错误（需要两列 T,Sa）: {spec_path}")

    T = data[:, 0]
    Sa = data[:, 1]
    ok = np.isfinite(T) & np.isfinite(Sa)
    T = T[ok]
    Sa = Sa[ok]
    if T.size < 2:
        raise ValueError(f"反应谱有效点不足: {spec_path}")

    order = np.argsort(T)
    T = T[order]
    Sa = Sa[order]

    if t1 < T.min() or t1 > T.max():
        raise ValueError(f"T1={t1:.4f} 超出反应谱范围 [{T.min():.4f}, {T.max():.4f}]")

    sa_t1 = float(np.interp(t1, T, Sa))
    if not np.isfinite(sa_t1) or sa_t1 <= 0:
        raise ValueError(f"插值得到的 SaMCE(T1) 无效: {sa_t1}")
    return sa_t1


def _resolve_ida_record_path(project_root: Path, model: str, temp: str, record: str) -> Path:
    normalized_model = MODEL_ALIASES.get(model, model)
    path = project_root / "Output_data" / f"MC8_{normalized_model}_{temp}" / "MC8_IDA_data_frag" / record
    if path.exists():
        return path

    raise FileNotFoundError(
        f"未找到 IDA 数据文件: {path}\n"
        f"当前 model={model!r} 已按 {normalized_model!r} 解析；请检查模型名、温度或结果文件是否已生成。"
    )


def plot_ida(
    file_paths: List[Union[str, Path]],
    labels: Optional[List[str]] = None,
    *,
    sheet: Union[str, int] = 0,
    spaghetti: bool = False,
    selected: Optional[List[int]] = None,
    show_quantiles: Tuple[int, ...] = (16, 50, 84),
    collapse_limit: Optional[float] = None,
    normalize_y: bool = False,
    sa_mce_t1: Optional[Union[float, List[float], Tuple[float, ...]]] = None,
    auto_sa_mce_from_pushover: bool = False,
    mce_spec_path: Optional[Union[str, Path]] = None,
    x_max: Optional[float] = None,
    density: int = 800,

    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    Xlim: Tuple[float, float] = None,
    Ylim: Tuple[float, float] = None,
    Yticks: Optional[Tuple[float, ...]] = None,
    save_path: str = None
):
    """
    绘制一组 IDA 曲线，并可叠加分位曲线（16/50/84）。

    参数说明：
    - file_paths: 每个工况对应的 Excel 文件路径列表。每个文件应包含 [DM1, IM1, DM2, IM2, ...] 列对。
    - labels: 图例标签列表，长度应与 file_paths 一致；为 None 时自动生成为 Case 1/2/...
    - sheet: 读取的工作表名或索引（默认 0）。
    - spaghetti: 是否绘制所有原始 IDA 曲线（灰色细线），用于查看离散性。
    - selected: 要高亮的曲线序号列表（从 0 开始），如 [0, 1] 表示第 1、2 条曲线。
    - show_quantiles: 需要绘制的分位线，可选 16/50/84；空元组 () 表示不绘制分位线。
    - collapse_limit: 坍塌截断阈值（按 DM）。设置后，DM 达到该值时该曲线后续点会被截断。
    - normalize_y: 是否将 y 轴归一化为 Sa(T1)/SaMCE(T1)。
    - sa_mce_t1: 手动指定 SaMCE(T1)。可传单个正数（所有文件共用）或与 file_paths 等长的正数列表。
    - auto_sa_mce_from_pushover: 为 True 且 sa_mce_t1=None 时，自动从模型输出读取 T1，再由 MCE 反应谱插值计算 SaMCE(T1)。
    - mce_spec_path: MCE 反应谱文件路径（两列：T, Sa）。None 时默认使用项目根目录下 Spectrum/MCE Level Spectrum.txt。
    - x_max: 统计分位线时的最大 DM 范围。仅影响分位曲线，不影响 raw 曲线绘制。
    - density: 分位曲线插值网格密度（点数），越大越平滑，计算也稍慢。
    - title: 图标题；None 表示不设置标题。
    - xlabel: x 轴名称。None 时根据文件名自动推断；IDR/RIDR 自动显示为百分比，PFA 不做百分比缩放。
    - ylabel: y 轴名称。None 时自动设置：未归一化为 "$Sa(T_1)$ [g]"，归一化为 "$Sa(T_1)/Sa_{MCE}(T_1)$"。
    - xlim: x 轴范围，如 (0, 0.2)；None 表示自动范围。
    - ylim: y 轴范围，如 (0, 2.0)；None 表示自动范围。
    """
    # ====== labels ======
    if labels is None:
        labels = [f"Case {i+1}" for i in range(len(file_paths))]

    dm_name = _infer_dm_name(file_paths, xlabel)
    x_scale = _dm_axis_scale(dm_name)
    display_xlabel = _format_xlabel(dm_name, xlabel, x_scale)
    display_xlim = _scale_axis_limits(Xlim, x_scale)

    # ====== y normalization ======
    if normalize_y:
        if sa_mce_t1 is not None:
            if np.isscalar(sa_mce_t1):
                norm_vals = [float(sa_mce_t1)] * len(file_paths)
            else:
                norm_vals = [float(v) for v in sa_mce_t1]
                if len(norm_vals) != len(file_paths):
                    raise ValueError("sa_mce_t1 为序列时，其长度必须与 file_paths 一致。")
            if not np.all(np.isfinite(norm_vals)) or np.any(np.asarray(norm_vals) <= 0):
                raise ValueError("sa_mce_t1 必须为有限正数。")
        elif auto_sa_mce_from_pushover:
            project_root = Path(__file__).resolve().parents[2]
            spec_file = Path(mce_spec_path) if mce_spec_path is not None else (project_root / "Spectrum" / "MCE Level Spectrum.txt")
            if not spec_file.exists():
                raise FileNotFoundError(f"未找到 MCE 反应谱文件: {spec_file}")
            norm_vals = []
            for fp in file_paths:
                model_dir = _infer_model_dir_from_ida_file(fp)
                t1 = _read_t1_from_model_dir(model_dir)
                norm_vals.append(_interp_sa_at_t1(spec_file, t1))
        else:
            raise ValueError("normalize_y=True 时，请提供 sa_mce_t1，或设置 auto_sa_mce_from_pushover=True。")
    else:
        norm_vals = [1.0] * len(file_paths)

    # ====== figure format ======
    fig, ax = plt.subplots(figsize=(8, 6))


    # file-level colors
    if len(file_paths) == 1:
        file_colors = ['blue']
    else:
        file_colors = cm.coolwarm(np.linspace(0, 1, len(file_paths)))

    for i, fp in enumerate(file_paths):
        y_norm = norm_vals[i]

        # ===== 1) raw data for spaghetti / selected =====
        fp = Path(fp)
        df_raw = pd.read_excel(fp, sheet_name=sheet, header=None).apply(pd.to_numeric, errors="coerce")
        df_raw = df_raw.dropna(axis=1, how="all")
        arr_raw = df_raw.to_numpy()

        if arr_raw.shape[1] % 2 != 0 and arr_raw.shape[1] >= 3 and _is_index_like(arr_raw[:, 0]):
            arr_raw = arr_raw[:, 1:]

        if arr_raw.shape[1] % 2 != 0:
            raise ValueError(f"Raw sheet columns must be even after dropping index-like col. got {arr_raw.shape[1]}")

        n_raw = arr_raw.shape[1] // 2

        def _get_raw_curve(k: int):
            dm = arr_raw[:, 2 * k]
            im = arr_raw[:, 2 * k + 1]
            ok = np.isfinite(dm) & np.isfinite(im) & (dm > 0) & (im > 0)
            return dm[ok] * x_scale, im[ok]

        # ===== 2) cleaned pairs for quantiles =====
        pairs = read_ida_pairs(fp, sheet=sheet, collapse_limit=collapse_limit)
        if y_norm != 1.0:
            pairs = [(dm, im / y_norm) for dm, im in pairs]

        # A) spaghetti using raw curves
        if spaghetti:
            for k in range(n_raw):
                dm, im = _get_raw_curve(k)
                if dm.size >= 2:
                    ax.plot(dm, im / y_norm, color="0.5", alpha=0.7, lw=0.8, label="_nolegend_")
                    ax.scatter(dm, im / y_norm, color="0.5", alpha=0.7, s=10, label="_nolegend_")

        # B) selected curves using raw curves
        if selected:
            cmap_sel = plt.get_cmap("viridis")
            cols = cmap_sel(np.linspace(0.2, 0.9, len(selected)))
            for c, idx in zip(cols, selected):
                if 0 <= idx < n_raw:
                    dm, im = _get_raw_curve(idx)
                    if dm.size >= 2:
                        ax.plot(
                            dm,
                            im / y_norm,
                            color=c,
                            lw=2.2,
                            marker="o",
                            ms=3,
                            mfc="none",
                            mew=1.2,
                            label=f"{labels[i]} - IDA #{idx+1}",
                        )

        # C) quantiles using cleaned pairs
        if show_quantiles:
            x, q16, q50, q84 = quantiles_on_grid(pairs, density=density, x_max=x_max)
            x = x * x_scale
            col = file_colors[i]

            if 50 in show_quantiles:
                ax.plot(
                    x,
                    q50,
                    color=col,
                    lw=2.6,
                    label=(labels[i] if len(file_paths) > 1 else "Median"),
                )
                # ax.scatter(
                #     x,
                #     q50,
                #     color=col,
                #     s=50,
                #     edgecolor="white",
                #     lw=0.8,
                #     label="_nolegend_",
                # )
            if 16 in show_quantiles or 84 in show_quantiles:
                leg = "16th/84th" if (len(file_paths) == 1 and 16 in show_quantiles) else "_nolegend_"
                if 16 in show_quantiles:
                    ax.plot(x, q16, color=col, lw=1.8, ls="--", label=leg)
                if 84 in show_quantiles:
                    ax.plot(x, q84, color=col, lw=1.8, ls="--", label="_nolegend_")

    # ============ Layout ============
    ax = plt.gca()

    # tick: inward + big font
    ax.tick_params(axis='both', direction='in', which='both', labelsize=25)

    # labels
    plt.xlabel(display_xlabel, fontsize=30, labelpad=12)
    plt.ylabel(ylabel, fontsize=30, labelpad=12)

    # title
    if title:
        plt.title(title, fontsize=18, pad=10)

    # grid（你如果不想要网格就改成 False）
    plt.grid(True, alpha=0.25)

    # limits
    if display_xlim is not None:
        plt.xlim(left=display_xlim[0] if display_xlim[0] is not None else None,
                right=display_xlim[1] if display_xlim[1] is not None else None)
    else:
        plt.xlim(left=0)

    if Ylim is not None:
        plt.ylim(bottom=Ylim[0] if Ylim[0] is not None else None,
                top=Ylim[1] if Ylim[1] is not None else None)
    else:
        plt.ylim(bottom=0)

    if Yticks is not None:
        plt.yticks(Yticks)

    # de-duplicate legend
    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    if by_label:
        # ✅ legend 左上（upper left）+ 字号
        plt.legend(by_label.values(), by_label.keys(),
                fontsize=18, loc='upper left', frameon=True, ncol=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]
    model = "PFSDF"
    temp = "40"
    DM = "PFA"
    record = f"IDA曲线_{DM}.xlsx"

    single_path = [
        _resolve_ida_record_path(project_root, model, temp, record)
    ]

    # 1) 同一温度下：画全部曲线 + 16/50/84 分位线（可开启归一化）
    # plot_ida(
    # file_paths=single_path,
    # labels=[f"{model} {temp}°C"],
    # sheet=0,
    # spaghetti=True,
    # selected=None,
    # show_quantiles=(16, 50, 84),
    # collapse_limit=None,

    # normalize_y=False,                  # 开启后 y 轴为 Sa(T1)/SaMCE(T1)
    # auto_sa_mce_from_pushover=True,    # 自动读取 MC8_PO_out/周期(s).out 并由 MCE 谱插值 SaMCE(T1)
    
    # x_max=None,       # 或者 0.2 / 0.05 等
    # density=300,
    # title=f"IDA Curves with Quantiles - {model} at {temp}°C",
    # Xlim=(0, 10),
    # )

    # 2) 同一温度：只画选中的若干条 IDA 曲线
    # plot_ida(
    # file_paths=single_path,
    # labels=[f"{model} {temp}°C"],
    # sheet=0,
    # spaghetti=False,
    # selected=[7,22],
    # show_quantiles=(),   # 关键：空元组 => 不画分位线
    # title=f"Selected IDA Curves - {model} at {temp}°C",
    # xlim=(0, 0.2),)


    # 3) 多温度对比：只画 50% 分位线（Median）
    temps = ["-20", "0", "20", "40"]
    temps = ["-20","-10" ,"0","10", "20","30", "40"]
    paths = [
        _resolve_ida_record_path(project_root, model, t, record)
        for t in temps
    ]
    labels = [f"{t}°C" for t in temps]

    plot_ida(
        file_paths=paths,
        labels=labels,
        sheet=0,
        spaghetti=False,
        selected=None,
        show_quantiles=(50,),
        x_max=None,
        density=150,
        Xlim=(0, 2),
        Ylim=(0, 1.5),
        Yticks=(0, 0.5, 1.0, 1.5),
        ylabel="$Sa (T_1)$",
        # ylabel="$Sa(T_1)/Sa_{MCE}(T_1)$",
        # save_path=project_root / "Paint" / f"IDA_{DM}_{model}.png"
        )
