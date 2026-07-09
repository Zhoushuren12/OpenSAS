from re import X
from tkinter import Y
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Global font setup
plt.rc("font", family="Times New Roman")
plt.rc("mathtext", fontset="stix")


@dataclass
class QuantileResult:
    """Cached quantile result for inverse IDA (DM -> IM)."""
    dm_name: str
    dm_axis_scale: float
    pct_x_dm: np.ndarray
    pct_16_im: np.ndarray
    pct_50_im: np.ndarray
    pct_84_im: np.ndarray
    Sa_MCE: Optional[float]
    normalize: bool
    collapse_limit: Optional[float]


class IDAPlotter:
    """
    IDA plotting/calculation helper:
    - get_y / get_percentile_line_x
    - compute_quantiles from raw Excel curves
    """

    def __init__(
        self,
        base_dir: Union[str, Path] = "Output_data",
        mce_spec_path: Optional[Union[str, Path]] = None,
        T0: float = 2.148,
    ):
        self.base_dir = Path(base_dir)
        self.mce_spec_path = Path(mce_spec_path) if mce_spec_path else None
        self.T0 = float(T0)

        plt.rc("font", family="Times New Roman")
        plt.rc("mathtext", fontset="stix")

        # cache key = (file_path, sheet, dm_upper, collapse_limit, x_max_for_quantiles, normalize, density)
        self._cache: Dict[Tuple[str, str, str, Optional[float], Optional[float], bool, int], QuantileResult] = {}

    # -----------------------
    # Basic helpers
    # -----------------------
    @staticmethod
    def get_x(y: List[float], x: List[float], y0: float, error: bool = True) -> Optional[float]:
        if y0 < min(y):
            return None if not error else (_ for _ in ()).throw(ValueError(f"y0 < min(y) ({y0} < {min(y)})"))
        if y0 > max(y):
            return None if not error else (_ for _ in ()).throw(ValueError(f"y0 > max(y) ({y0} > {max(y)})"))

        for i in range(len(y) - 1):
            if y[i] == y0:
                return x[i]
            if y[i] < y0 <= y[i + 1]:
                k = (x[i + 1] - x[i]) / (y[i + 1] - y[i])
                return k * (y0 - y[i]) + x[i]
        return None if not error else (_ for _ in ()).throw(ValueError("Intersection not found"))

    @staticmethod
    def get_y(x: List[float], y: List[float], x0: float, error: bool = True) -> Optional[float]:
        """Linear interpolation of y at x=x0."""
        if x0 < min(x):
            return None if not error else (_ for _ in ()).throw(ValueError(f"x0 < min(x) ({x0} < {min(x)})"))
        if x0 > max(x):
            return None if not error else (_ for _ in ()).throw(ValueError(f"x0 > max(x) ({x0} > {max(x)})"))

        for i in range(len(x) - 1):
            if x[i] == x0:
                return y[i]
            if x[i] < x0 <= x[i + 1]:
                k = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                return k * (x0 - x[i]) + y[i]
        return None if not error else (_ for _ in ()).throw(ValueError("Intersection not found"))

    def get_percentile_line_x(
        self,
        all_x: List[List[float]],
        all_y: List[List[float]],
        p: float,
        n: int = 300,
        x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantile line for a curve family: fix x grid, take percentile of y.
        """
        if x is None:
            x1 = min(min(i) for i in all_x)
            x2 = max(max(i) for i in all_x)
            x = np.linspace(x1, x2, n)

        y_out = []
        for xi in x:
            yi = []
            for line_x, line_y in zip(all_x, all_y):
                res = self.get_y(line_x, line_y, float(xi), error=False)
                if res is not None and np.isfinite(res):
                    yi.append(res)

            if len(yi) == 0:
                y_out.append(np.nan)
            else:
                y_out.append(np.percentile(yi, p))

        return np.asarray(x, dtype=float), np.asarray(y_out, dtype=float)

    def _get_Sa_MCE(self) -> Optional[float]:
        """Read MCE spectrum and interpolate Sa(T0); return None if missing."""
        if self.mce_spec_path is None:
            return None
        try:
            data = np.loadtxt(self.mce_spec_path)
            T = data[:, 0]
            Sa = data[:, 1]
            return float(self.get_y(list(T), list(Sa), self.T0, error=True))
        except Exception:
            return None

    @staticmethod
    def _dm_axis_scale(dm_upper: str) -> float:
        return 100.0 if dm_upper in {"IDR", "RIDR"} else 1.0

    @staticmethod
    def _is_index_like(col: np.ndarray) -> bool:
        col = col[~np.isnan(col)]
        if col.size < 2:
            return False
        if not np.all(np.isfinite(col)):
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

    @classmethod
    def _read_ida_sheet(cls, file_path: Path, sheet_name: Union[str, int]) -> np.ndarray:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None).apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")
        arr = df.to_numpy()
        ncols = arr.shape[1]
        if ncols % 2 != 0:
            if ncols >= 3 and cls._is_index_like(arr[:, 0]):
                warnings.warn(
                    f"Detected index-like first column in '{sheet_name}'; dropping it to form DM/IM pairs.",
                    RuntimeWarning,
                )
                arr = arr[:, 1:]
                ncols = arr.shape[1]
        if ncols % 2 != 0:
            raise ValueError(f"'{sheet_name}' columns must be even (DM/IM pairs). current={ncols}")
        return arr

    # -----------------------
    # Build curve families from Excel
    # -----------------------
    def _build_lines_from_excel(
        self,
        file_path: Path,
        sheet_name: Union[str, int],
        collapse_limit: Optional[float],
    ) -> Tuple[List[List[float]], List[List[float]], str]:
        """
        Read a sheet of [DM1, IM1, DM2, IM2, ...] and build DM/IM lines.
        - drop NaN / <0
        - sort by DM (x-axis)
        - apply collapse_limit (keep first exceeding point)
        - prepend origin
        """
        arr = self._read_ida_sheet(file_path, sheet_name=sheet_name)
        ncols = arr.shape[1]

        stem = file_path.stem
        if "IDA" in stem and "_" in stem:
            dm_name = stem.split("_", 1)[1]
        elif "_" in stem:
            dm_name = stem.split("_", 1)[1]
        else:
            dm_name = stem
        dm_upper = dm_name.upper()

        num_curves = ncols // 2
        DM_lines: List[List[float]] = []
        IM_lines: List[List[float]] = []

        for i in range(num_curves):
            DM_i = arr[:, 2 * i]
            IM_i = arr[:, 2 * i + 1]
            valid = (~np.isnan(DM_i)) & (~np.isnan(IM_i)) & (DM_i >= 0) & (IM_i >= 0)
            DM_i = DM_i[valid]
            IM_i = IM_i[valid]
            if DM_i.size < 2:
                continue

            # sort by DM (x-axis)
            order = np.argsort(DM_i)
            DM_i = DM_i[order]
            IM_i = IM_i[order]

            # collapse_limit (keep first exceed)
            if collapse_limit is not None:
                exceed = DM_i >= float(collapse_limit)
                if np.any(exceed):
                    k = int(np.argmax(exceed))
                    IM_i = IM_i[: k + 1]
                    DM_i = DM_i[: k + 1]

            DM_lines.append([0.0] + list(map(float, DM_i)))
            IM_lines.append([0.0] + list(map(float, IM_i)))

        if len(DM_lines) < 2:
            raise ValueError("Not enough valid curves to compute quantiles (need >= 2).")

        return DM_lines, IM_lines, dm_upper

    # -----------------------
    # Quantile computation (cached)
    # -----------------------
    def compute_quantiles(
        self,
        file_path: Union[str, Path],
        *,
        raw_sheet_name: Union[str, int] = 0,
        collapse_limit: Optional[float] = None,
        x_max_for_quantiles: Optional[float] = 5.0,
        normalize: bool = False,
        density: int = 3000,
        use_cache: bool = True,
    ) -> QuantileResult:
        file_path = Path(file_path)
        DM_lines, IM_lines, dm_upper = self._build_lines_from_excel(
            file_path, sheet_name=raw_sheet_name, collapse_limit=collapse_limit
        )

        key = (
            str(file_path.resolve()),
            str(raw_sheet_name),
            dm_upper,
            collapse_limit,
            x_max_for_quantiles,
            bool(normalize),
            int(density),
        )
        if use_cache and key in self._cache:
            return self._cache[key]

        Sa_MCE = self._get_Sa_MCE() if normalize else None
        if normalize and (Sa_MCE is None or Sa_MCE == 0):
            raise ValueError("normalize=True but Sa_MCE could not be determined")

        if x_max_for_quantiles is not None:
            x_max = float(x_max_for_quantiles)
            filtered_dm: List[List[float]] = []
            filtered_im: List[List[float]] = []
            for dm_line, im_line in zip(DM_lines, IM_lines):
                dm_arr = np.asarray(dm_line, dtype=float)
                im_arr = np.asarray(im_line, dtype=float)
                keep = dm_arr <= x_max
                dm_keep = dm_arr[keep]
                im_keep = im_arr[keep]
                if dm_keep.size >= 2:
                    filtered_dm.append(dm_keep.tolist())
                    filtered_im.append(im_keep.tolist())
            if len(filtered_dm) >= 2:
                DM_lines, IM_lines = filtered_dm, filtered_im
            else:
                warnings.warn(
                    "x_max_for_quantiles removed too many points; using original data for quantile lines.",
                    RuntimeWarning,
                )

        pct_x, pct_16 = self.get_percentile_line_x(DM_lines, IM_lines, p=16, n=density)
        _, pct_50 = self.get_percentile_line_x(DM_lines, IM_lines, p=50, n=density, x=pct_x)
        _, pct_84 = self.get_percentile_line_x(DM_lines, IM_lines, p=84, n=density, x=pct_x)

        if normalize:
            pct_16 = pct_16 / Sa_MCE
            pct_50 = pct_50 / Sa_MCE
            pct_84 = pct_84 / Sa_MCE

        res = QuantileResult(
            dm_name=dm_upper,
            dm_axis_scale=self._dm_axis_scale(dm_upper),
            pct_x_dm=pct_x,
            pct_16_im=pct_16,
            pct_50_im=pct_50,
            pct_84_im=pct_84,
            Sa_MCE=Sa_MCE,
            normalize=normalize,
            collapse_limit=collapse_limit,
        )
        if use_cache:
            self._cache[key] = res
        return res


_IDA_HELPER = IDAPlotter()


def compute_quantile_lines_from_raw(
    file_path: Union[str, Path],
    raw_sheet_name: Union[str, int] = 0,
    quantiles: Tuple[int, ...] = (16, 50, 84),
    collapse_limit: Optional[float] = None,
    x_max_for_quantiles: Optional[float] = 5.0,
    density: int = 100,
    use_cache: bool = True,
) -> dict:
    """
    Compute quantile lines (16/50/84) from a raw sheet.
    Returns: {"x": pct_x, 16: pct_16, 50: pct_50, 84: pct_84}
    """
    file_path = Path(file_path)
    q = _IDA_HELPER.compute_quantiles(
        file_path,
        raw_sheet_name=raw_sheet_name,
        collapse_limit=collapse_limit,
        x_max_for_quantiles=x_max_for_quantiles,
        density=density,
        use_cache=use_cache,
        normalize=False,
    )

    res = {"x": q.pct_x_dm, 16: q.pct_16_im, 50: q.pct_50_im, 84: q.pct_84_im}
    return res


# ============================================================
#                         Plot function
# ============================================================

def plot_IDA_advanced(
    file_paths: list,
    labels: list = None,

    # ========= Raw curve selection =========
    single_curve_index: Optional[Union[int, List[int]]] = None,
    single_curve_only: bool = True,
    single_curve_style: Optional[dict] = None,
    single_curve_cmap: str = "viridis",
    # ==================================

    # spaghetti
    plot_individual: bool = False,
    raw_sheet_name: Union[str, int] = 0,

    # quantiles
    quantiles: tuple = (16, 50, 84),
    compute_quantiles: bool = True,
    collapse_limit: Optional[float] = None,
    x_max_for_quantiles: Optional[float] = 5.0,
    density: int = 3000,
    use_cache: bool = True,

    title: str = None,
    xlabel: str = 'IDR',
    ylabel: str = r'$Sa(T_1)$ [g]',
    Xlim: Tuple[float, float] = None,
    Ylim: Tuple[float, float] = None,
    save_path: str = None
):
    """Main IDA plotting function."""
    if labels is None:
        labels = [f"Case {i+1}" for i in range(len(file_paths))]

    linestyle_map = {16: '--', 50: '-', 84: '--'}
    plt.figure(figsize=(8, 6))

    # file-level colors
    if len(file_paths) == 1:
        file_colors = ['blue']
    else:
        file_colors = cm.coolwarm(np.linspace(0, 1, len(file_paths)))

    # default single-curve style
    if single_curve_style is None:
        single_curve_style = dict(
            linewidth=2.2, linestyle='-', marker='o',
            markersize=4, markerfacecolor='none', markeredgewidth=1.5
        )

    # normalize single_curve_index
    if single_curve_index is not None:
        if isinstance(single_curve_index, int):
            single_curve_index = [single_curve_index]
        elif isinstance(single_curve_index, (list, tuple, np.ndarray)):
            single_curve_index = list(single_curve_index)
        else:
            raise TypeError("single_curve_index must be int, list[int], or None")

    cmap_obj = plt.get_cmap(single_curve_cmap)

    for i, fpath in enumerate(file_paths):
        path_obj = Path(fpath)
        if not path_obj.exists():
            print(f"[WARN] File not found: {fpath}")
            continue

        color_file = file_colors[i]
        label_base = labels[i]

        try:
            # read raw data only when needed
            need_raw = plot_individual or (single_curve_index is not None) or (compute_quantiles and quantiles)
            arr_raw = None
            num_curves = 0

            if need_raw:
                arr_raw = IDAPlotter._read_ida_sheet(path_obj, sheet_name=raw_sheet_name)
                num_curves = arr_raw.shape[1] // 2
                if num_curves <= 0:
                    raise ValueError(f"Raw sheet '{raw_sheet_name}' has no (IDR, Sa) column pairs.")

            # =================================================
            # A) Selected raw curves
            # =================================================
            if single_curve_index is not None:
                curve_colors = cmap_obj(np.linspace(0.15, 0.90, len(single_curve_index)))

                for j, idx in enumerate(single_curve_index):
                    if idx < 0 or idx >= num_curves:
                        print(f"[WARN] single_curve_index={idx} out of range (Total {num_curves}). Skipping.")
                        continue

                    idr = arr_raw[:, 2 * idx]
                    sa = arr_raw[:, 2 * idx + 1]
                    valid = np.isfinite(idr) & np.isfinite(sa) & (idr > 0) & (sa > 0)

                    plt.plot(
                        idr[valid], sa[valid],
                        color=curve_colors[j],
                        zorder=20,
                        label=f"{label_base} - IDA #{idx+1}",
                        **single_curve_style
                    )

                if single_curve_only:
                    continue

            # =================================================
            # B) Spaghetti: all individual curves
            # =================================================
            if plot_individual:
                for k in range(num_curves):
                    raw_idr = arr_raw[:, 2 * k]
                    raw_sa = arr_raw[:, 2 * k + 1]
                    valid = np.isfinite(raw_idr) & np.isfinite(raw_sa) & (raw_idr > 0) & (raw_sa > 0)

                    lbl = 'Individual' if (i == 0 and k == 0) else "_nolegend_"

                    plt.plot(
                        raw_idr[valid], raw_sa[valid],
                        color='gray', alpha=0.8, linewidth=0.5,
                        marker='o', markersize=3, markerfacecolor='pink', markeredgewidth=0,
                        zorder=1, label=lbl
                    )

            # =================================================
            # C) Quantile lines
            # =================================================
            if compute_quantiles and quantiles:
                frac = compute_quantile_lines_from_raw(
                    file_path=fpath,
                    raw_sheet_name=raw_sheet_name,
                    quantiles=tuple(quantiles),
                    collapse_limit=collapse_limit,
                    x_max_for_quantiles=x_max_for_quantiles,
                    density=density,
                    use_cache=use_cache,
                )

                for q in quantiles:
                    if q not in (16, 50, 84):
                        continue

                    # legend rules
                    lbl = None
                    if len(file_paths) == 1:
                        if q == 50:
                            lbl = "Median"
                        elif q == 16:
                            lbl = "16th/84th"
                    else:
                        if q == 50:
                            lbl = label_base

                    plt.plot(
                        frac["x"], frac[q],
                        color=color_file,
                        linestyle=linestyle_map[q],
                        linewidth=2.5 if q == 50 else 1.5,
                        marker='o',
                        markersize=2,        
                        markerfacecolor='white', 
                        zorder=10,
                        label=lbl
                        
                    )

        except Exception as e:
            print(f"[ERROR] Processing {path_obj.name}: {e}")
            import traceback
            traceback.print_exc()

    # ============ Layout ============
    ax = plt.gca()

    # tick: inward + big font
    ax.tick_params(axis='both', direction='in', which='both', labelsize=18)

    # labels
    plt.xlabel(xlabel, fontsize=25, labelpad=12)
    plt.ylabel(ylabel, fontsize=25, labelpad=12)

    # title
    if title:
        plt.title(title, fontsize=18, pad=10)

    plt.grid(True, alpha=0.25)

    # limits
    if Xlim is not None:
        plt.xlim(left=Xlim[0] if Xlim[0] is not None else None,
                right=Xlim[1] if Xlim[1] is not None else None)
    else:
        plt.xlim(left=0)

    if Ylim is not None:
        plt.ylim(bottom=Ylim[0] if Ylim[0] is not None else None,
                top=Ylim[1] if Ylim[1] is not None else None)
    else:
        plt.ylim(bottom=0)

    # de-duplicate legend
    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(),
                fontsize=18, loc='upper left', frameon=True, ncol=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

if __name__ == "__main__":
    model = "PFSDF"
    temp = "10"
    record = "IDA鏇茬嚎_IDR.xlsx"

    project_root = Path(__file__).resolve().parents[2]
    single_path = [
        project_root / "Output_data" / f"MC8_{model}_{temp}" / "MC8_IDA_data_frag" / record
    ]

    # # 1) 鍚屼竴娓╁害涓嬶紝鐢荤 1/2 鏉″師濮婭DA鏇茬嚎锛坕dx=0/1锛夛紝骞跺尯鍒嗛鑹诧紙鍙敾杩欎袱鏉★級
    # plot_IDA_advanced(
    #     file_paths=single_path,
    #     labels=[f"{model} {temp}掳C"],
    #     single_curve_index=[7,22],
    #     single_curve_only=True,
    #     single_curve_cmap="viridis",
    #     raw_sheet_name=0,

    #     compute_quantiles=True)

    # plot_IDA_advanced(
    #     file_paths=single_path,
    #     labels=[f"{model} {temp}掳C"],
    #     single_curve_only=True,     # 杩欓噷 single_curve_index=None锛屾墍浠ヨ鍙傛暟涓嶄細褰卞搷
    #     raw_sheet_name=0,
    #     plot_individual=True,
    #     quantiles=[16, 50, 84],

    #     # 鍙皟鍙傛暟
    #     density=500,
    #     use_cache=True,
    #     title=f"IDA Curves with Quantiles - {model} at {temp}掳C",
    #     Xlim=(0, 0.1)
    # )

    temps = ["-20", "0", "20", "40"]
    temps = ["-20","-10" ,"0","10", "20","30", "40"]
    labels = [f"{t}掳C" for t in temps]
    all_paths = [
        project_root / "Output_data" / f"MC8_{model}_{t}" / "MC8_IDA_data_frag" / record
        for t in temps
    ]

    plot_IDA_advanced(
        file_paths=all_paths,
        labels=labels,
        plot_individual=False,
        quantiles=[50],
        compute_quantiles=True,
        raw_sheet_name=0,
        x_max_for_quantiles=5.0,
        density=300,
        Xlim=(0, 0.1),
        Ylim=(0, 2),
    )
