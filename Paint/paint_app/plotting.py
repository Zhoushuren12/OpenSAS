"""Unified Matplotlib plotting service used by the desktop UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.figure import Figure

from .catalog import AnalysisCase


CELSIUS = chr(0x2103)

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "SimSun", "DejaVu Serif"]
mpl.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.linewidth"] = 1.2


def normalize_temperature_label(text: object) -> str:
    value = str(text)
    for old in (
        "degC",
        "DegC",
        "degree C",
        "degrees C",
        "°C",
        r"$^\circ$C",
        r"$\degree C$",
        r"\degree C",
    ):
        value = value.replace(old, CELSIUS)
    return value


def _label_font(text: object) -> str:
    value = normalize_temperature_label(text)
    return "SimSun" if CELSIUS in value or re.search(r"[\u4e00-\u9fff]", value) else "Times New Roman"


def _set_axis_label(axis, which: str, text: object) -> None:
    value = normalize_temperature_label(text)
    setter = axis.set_xlabel if which == "x" else axis.set_ylabel
    setter(value, fontsize=25, fontname=_label_font(value), labelpad=10)


TH_METRIC_FILES = {
    "IDR": "层间位移角.csv",
    "RIDR": "残余层间位移角.csv",
    "CIDR": "累积层间位移角.csv",
    "PFA": "层加速度(g).csv",
    "PFV": "层速度(mm_s).csv",
    "SHEAR": "楼层剪力(kN).csv",
    "DCF": "DCF.csv",
}

TH_RECORD_FILES = {
    "IDR": "层间位移角.out",
    "RIDR": "残余层间位移角.out",
    "CIDR": "累积层间位移角.out",
    "PFA": "层加速度(g).out",
    "PFV": "层速度.out",
    "SHEAR": "楼层剪力(kN).out",
    "DCF": "DCF.out",
}

METRIC_LABELS = {
    "IDR": "IDR (%)",
    "RIDR": "RIDR (%)",
    "CIDR": "CIDR (%)",
    "PFA": "PFA (g)",
    "PFV": "Floor velocity (mm/s)",
    "SHEAR": "Story shear (kN)",
    "DCF": "DCF",
}


@dataclass(frozen=True)
class PlotRequest:
    analysis: str
    plot_type: str
    cases: tuple[AnalysisCase, ...]
    level: str = ""
    metric: str = "IDR"
    statistic: str = "median"
    record: str = ""
    story: int = 1
    ds: int = 1
    normalize: bool = False
    title: str = ""
    grid: bool = True
    sa_value: float = 1.0


def _metric_scale(metric: str) -> float:
    return 100.0 if metric.upper() in {"IDR", "RIDR", "CIDR"} else 1.0


def _numeric_frame(path: Path, *, index_col: int | None = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        frame = pd.read_csv(stream, index_col=index_col)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame.dropna(axis=0, how="all").dropna(axis=1, how="all")


def _load_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.asarray(np.loadtxt(path), dtype=float)
    if values.ndim == 2:
        values = values[:, -1]
    return values.reshape(-1)


def _statistic(values: np.ndarray, name: str, axis: int = 1) -> np.ndarray:
    name = name.lower()
    absolute = np.abs(values)
    if name == "mean":
        return np.nanmean(absolute, axis=axis)
    if name == "p16":
        return np.nanpercentile(absolute, 16, axis=axis)
    if name == "p84":
        return np.nanpercentile(absolute, 84, axis=axis)
    if name == "max":
        return np.nanmax(absolute, axis=axis)
    return np.nanmedian(absolute, axis=axis)


def _floor_values(index: Iterable[object], count: int) -> np.ndarray:
    floors: list[float] = []
    for position, value in enumerate(index):
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        floors.append(float(match.group()) if match else float(position + 1))
    result = np.asarray(floors, dtype=float)
    if result.size != count or len(np.unique(result)) < max(1, result.size // 2):
        result = np.arange(1, count + 1, dtype=float)
    return result


class PlotService:
    """Create figures without opening standalone Matplotlib windows."""

    def create(self, request: PlotRequest) -> Figure:
        if not request.cases:
            raise ValueError("请至少选择一个工况。")
        dispatch = {
            "capacity_drift": self._plot_capacity,
            "capacity_displacement": self._plot_capacity,
            "capacity_normalized": self._plot_capacity,
            "response_profile": self._plot_response_profile,
            "response_boxplot": self._plot_response_boxplot,
            "record_profile": self._plot_record_profile,
            "time_history": self._plot_time_history,
            "hinge_profile": self._plot_hinge_profile,
            "ida_curves": self._plot_ida_curves,
            "ida_quantiles": self._plot_ida_quantiles,
            "fragility": self._plot_fragility,
            "fragility_surface": self._plot_fragility_surface,
            "psdm": self._plot_psdm,
            "exceedance": self._plot_exceedance,
            "convergence": self._plot_convergence,
        }
        try:
            figure = dispatch[request.plot_type](request)
        except KeyError as exc:
            raise ValueError(f"暂不支持的绘图类型：{request.plot_type}") from exc
        return figure

    @staticmethod
    def export(figure: Figure, path: str | Path, dpi: int = 300) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            layout_right = float(getattr(figure, "_paint_layout_right", 0.95))
            layout_bottom = float(getattr(figure, "_paint_layout_bottom", 0.01))
            figure.tight_layout(rect=[0.01, layout_bottom, layout_right, 0.95])
        except (ValueError, TypeError):
            pass
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        return output

    @staticmethod
    def _new_figure(*, projection: str | None = None) -> tuple[Figure, object]:
        figure = Figure(figsize=(10.0, 7.4), dpi=100)
        axis = figure.add_subplot(111, projection=projection)
        return figure, axis

    @staticmethod
    def _finish(axis, request: PlotRequest, *, legend: bool = True) -> None:
        if request.title.strip():
            title = normalize_temperature_label(request.title.strip())
            axis.set_title(title, fontsize=20, fontname=_label_font(title), pad=14)
        if request.grid and not hasattr(axis, "zaxis"):
            axis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.20)
        if not hasattr(axis, "zaxis"):
            axis.minorticks_on()
            axis.tick_params(axis="both", direction="in", which="major", labelsize=18, length=6, width=1.1)
            axis.tick_params(axis="both", direction="in", which="minor", length=3, width=0.8)
            for spine in axis.spines.values():
                spine.set_linewidth(1.2)
        layout_right = 0.95
        if legend:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                normalized = [normalize_temperature_label(label) for label in labels]
                if len(handles) > 4:
                    axis.legend(
                        handles,
                        normalized,
                        loc="center left",
                        bbox_to_anchor=(1.01, 0.5),
                        frameon=False,
                        fontsize=15,
                        handlelength=2.4,
                        labelspacing=0.55,
                    )
                    layout_right = 0.78
                else:
                    axis.legend(handles, normalized, loc="best", frameon=False, fontsize=15)
        try:
            axis.figure.tight_layout(rect=[0.01, 0.01, layout_right, 0.95])
        except (ValueError, TypeError):
            pass
        axis.figure._paint_layout_right = layout_right
        axis.figure._paint_layout_bottom = 0.01

    @staticmethod
    def _require_plotted(plotted: int, errors: list[str]) -> None:
        if plotted:
            return
        detail = "\n".join(errors[:5]) if errors else "没有找到符合条件的数据。"
        raise RuntimeError(f"没有可绘制的数据。\n{detail}")

    def _plot_capacity(self, request: PlotRequest) -> Figure:
        file_map = {
            "capacity_drift": ("屋顶位移角(%)-基底剪力(kN).txt", "Roof drift (%)", "Base shear (kN)"),
            "capacity_displacement": ("屋顶位移(mm)-基底剪力(kN).txt", "Roof displacement (mm)", "Base shear (kN)"),
            "capacity_normalized": ("屋顶位移角(%)-归一化基底剪力.txt", "Roof drift (%)", "Normalized base shear"),
        }
        file_name, xlabel, ylabel = file_map[request.plot_type]
        figure, axis = self._new_figure()
        plotted = 0
        errors: list[str] = []
        for case in request.cases:
            if case.po_dir is None:
                continue
            path = case.po_dir / file_name
            try:
                data = np.asarray(np.loadtxt(path), dtype=float)
                if data.ndim != 2 or data.shape[1] < 2:
                    raise ValueError("文件不是两列曲线数据")
                axis.plot(data[:, 0], data[:, 1], linewidth=2.0, label=case.display_name)
                plotted += 1
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        self._require_plotted(plotted, errors)
        _set_axis_label(axis, "x", xlabel)
        _set_axis_label(axis, "y", ylabel)
        self._finish(axis, request)
        return figure

    @staticmethod
    def _stats_dir(case: AnalysisCase, level: str) -> Path:
        root = case.th_dirs.get(level)
        if root is None:
            raise FileNotFoundError(f"{case.display_name} 没有 {level} 时程结果")
        canonical = root / "结果统计"
        if canonical.is_dir():
            return canonical
        alternative = next((item for item in root.iterdir() if item.is_dir() and any(item.glob("*.csv"))), None)
        if alternative is None:
            raise FileNotFoundError(f"未找到结果统计目录：{root}")
        return alternative

    def _plot_response_profile(self, request: PlotRequest) -> Figure:
        metric = request.metric.upper()
        file_name = TH_METRIC_FILES[metric]
        scale = _metric_scale(metric)
        figure, axis = self._new_figure()
        plotted = 0
        errors: list[str] = []
        for case in request.cases:
            try:
                frame = _numeric_frame(self._stats_dir(case, request.level) / file_name)
                values = _statistic(frame.to_numpy(dtype=float), request.statistic) * scale
                floors = _floor_values(frame.index, len(values))
                axis.plot(values, floors, marker="o", markersize=5.0, linewidth=2.0, label=case.display_name)
                plotted += 1
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        self._require_plotted(plotted, errors)
        _set_axis_label(axis, "x", METRIC_LABELS[metric])
        _set_axis_label(axis, "y", "Story")
        self._finish(axis, request)
        return figure

    def _plot_response_boxplot(self, request: PlotRequest) -> Figure:
        metric = request.metric.upper()
        file_name = TH_METRIC_FILES[metric]
        scale = _metric_scale(metric)
        figure, axis = self._new_figure()
        datasets: list[np.ndarray] = []
        labels: list[str] = []
        errors: list[str] = []
        for case in request.cases:
            try:
                frame = _numeric_frame(self._stats_dir(case, request.level) / file_name)
                data = np.abs(frame.to_numpy(dtype=float))
                per_record = np.nanmax(data, axis=0) * scale
                per_record = per_record[np.isfinite(per_record)]
                if per_record.size:
                    datasets.append(per_record)
                    labels.append(case.display_name)
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        self._require_plotted(len(datasets), errors)
        artists = axis.boxplot(datasets, tick_labels=labels, patch_artist=True, showmeans=True)
        colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        for index, box in enumerate(artists["boxes"]):
            box.set_facecolor(colors[index % len(colors)])
            box.set_alpha(0.72)
        _set_axis_label(axis, "y", METRIC_LABELS[metric])
        axis.tick_params(axis="x", rotation=25)
        self._finish(axis, request, legend=False)
        return figure

    def _plot_record_profile(self, request: PlotRequest) -> Figure:
        metric = request.metric.upper()
        file_name = TH_RECORD_FILES[metric]
        scale = _metric_scale(metric)
        figure, axis = self._new_figure()
        plotted = 0
        errors: list[str] = []
        for case in request.cases:
            root = case.th_dirs.get(request.level)
            if root is None:
                continue
            try:
                values = np.abs(_load_vector(root / request.record / file_name)) * scale
                first_floor = 0 if metric in {"IDR", "RIDR", "CIDR"} and values.size > 1 else 1
                floors = np.arange(first_floor, first_floor + values.size)
                axis.plot(values, floors, marker="o", markersize=5.0, linewidth=2.0, label=case.display_name)
                plotted += 1
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        self._require_plotted(plotted, errors)
        _set_axis_label(axis, "x", METRIC_LABELS[metric])
        _set_axis_label(axis, "y", "Story")
        self._finish(axis, request)
        return figure

    def _plot_time_history(self, request: PlotRequest) -> Figure:
        figure, axis = self._new_figure()
        plotted = 0
        errors: list[str] = []
        heights = [5500.0] + [4300.0] * 20
        for case in request.cases:
            root = case.th_raw_dirs.get(request.level)
            if root is None:
                errors.append(f"{case.display_name}: 缺少原始时程目录")
                continue
            record_dir = root / request.record
            try:
                top = _load_vector(record_dir / f"Disp{request.story + 1}.out")
                bottom = _load_vector(record_dir / f"Disp{request.story}.out")
                count = min(top.size, bottom.size)
                response = (top[:count] - bottom[:count]) / heights[request.story - 1] * 100.0
                time_path = record_dir / "Time.out"
                time = _load_vector(time_path)[:count] if time_path.exists() else np.arange(count) * 0.02
                count = min(time.size, response.size)
                axis.plot(time[:count], response[:count], linewidth=1.5, label=case.display_name)
                plotted += 1
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        self._require_plotted(plotted, errors)
        _set_axis_label(axis, "x", "Time (s)")
        _set_axis_label(axis, "y", f"Story {request.story} IDR (%)")
        self._finish(axis, request)
        return figure

    def _plot_hinge_profile(self, request: PlotRequest) -> Figure:
        kind_map = {"BEAM": "梁铰", "COLUMN": "柱铰", "PANEL": "节点域"}
        suffix_map = {"mean": "mean", "p16": "16th", "median": "50th", "p84": "84th", "max": "84th"}
        kind = kind_map.get(request.metric.upper(), "梁铰")
        suffix = suffix_map.get(request.statistic.lower(), "50th")
        figure, axis = self._new_figure()
        plotted = 0
        errors: list[str] = []
        for case in request.cases:
            try:
                path = self._stats_dir(case, request.level) / f"{kind}_统计_{suffix}.csv"
                frame = _numeric_frame(path)
                values = np.nanmax(np.abs(frame.to_numpy(dtype=float)), axis=1)
                floors = _floor_values(frame.index, len(values))
                grouped = pd.DataFrame({"floor": floors, "value": values}).groupby("floor", as_index=False)["value"].max()
                axis.plot(grouped["value"], grouped["floor"], marker="o", markersize=5.0, linewidth=2.0, label=case.display_name)
                plotted += 1
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        self._require_plotted(plotted, errors)
        hinge_labels = {"梁铰": "Beam hinge rotation", "柱铰": "Column hinge rotation", "节点域": "Panel-zone deformation"}
        _set_axis_label(axis, "x", hinge_labels[kind])
        _set_axis_label(axis, "y", "Story")
        self._finish(axis, request)
        return figure

    @staticmethod
    def _ida_workbook(case: AnalysisCase, prefix: str, metric: str) -> Path:
        if case.fragility_dir is None:
            raise FileNotFoundError(f"{case.display_name} 没有 IDA 后处理结果")
        return case.fragility_dir / f"{prefix}_{metric}.xlsx"

    @staticmethod
    def _read_ida_book(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not path.exists():
            raise FileNotFoundError(path)
        book = pd.ExcelFile(path)
        raw = pd.read_excel(path, sheet_name=book.sheet_names[0], header=None)
        quantiles = pd.read_excel(path, sheet_name=book.sheet_names[1], header=None)
        return raw, quantiles

    @staticmethod
    def _xy_pair(frame: pd.DataFrame, pair: int, *, start_row: int = 1) -> tuple[np.ndarray, np.ndarray]:
        x = pd.to_numeric(frame.iloc[start_row:, pair * 2], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame.iloc[start_row:, pair * 2 + 1], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        return x[valid], y[valid]

    def _plot_ida_curves(self, request: PlotRequest) -> Figure:
        metric = request.metric.upper()
        scale = _metric_scale(metric)
        figure, axis = self._new_figure()
        plotted = 0
        errors: list[str] = []
        colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        for case_index, case in enumerate(request.cases):
            try:
                raw, quantiles = self._read_ida_book(self._ida_workbook(case, "IDA曲线", metric))
                color = colors[case_index % len(colors)]
                for pair in range(raw.shape[1] // 2):
                    x, y = self._xy_pair(raw, pair)
                    if x.size:
                        axis.plot(x * scale, y, color=color, alpha=0.13, linewidth=0.7)
                x50, y50 = self._xy_pair(quantiles, 1)
                axis.plot(x50 * scale, y50, color=color, linewidth=2.0, label=f"{case.display_name} 50th")
                plotted += 1
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        self._require_plotted(plotted, errors)
        _set_axis_label(axis, "x", METRIC_LABELS.get(metric, metric))
        _set_axis_label(axis, "y", "Sa (g)")
        self._finish(axis, request)
        return figure

    def _plot_ida_quantiles(self, request: PlotRequest) -> Figure:
        metric = request.metric.upper()
        scale = _metric_scale(metric)
        figure, axis = self._new_figure()
        plotted = 0
        errors: list[str] = []
        styles = ["--", "-", ":"]
        labels = ["16th", "50th", "84th"]
        for case in request.cases:
            try:
                _, quantiles = self._read_ida_book(self._ida_workbook(case, "IDA曲线", metric))
                for pair in range(min(3, quantiles.shape[1] // 2)):
                    x, y = self._xy_pair(quantiles, pair)
                    axis.plot(x * scale, y, linestyle=styles[pair], linewidth=2.0, label=f"{case.display_name} {labels[pair]}")
                plotted += 1
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        self._require_plotted(plotted, errors)
        _set_axis_label(axis, "x", METRIC_LABELS.get(metric, metric))
        _set_axis_label(axis, "y", "Sa (g)")
        self._finish(axis, request)
        return figure

    @staticmethod
    def _fragility_xy(case: AnalysisCase, metric: str, ds: int) -> tuple[np.ndarray, np.ndarray]:
        path = PlotService._ida_workbook(case, "易损性曲线", metric)
        frame = pd.read_excel(path, sheet_name=0, header=None)
        x = pd.to_numeric(frame.iloc[2:, 0], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame.iloc[2:, ds], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        order = np.argsort(x[valid])
        return x[valid][order], y[valid][order]

    def _plot_fragility(self, request: PlotRequest) -> Figure:
        metric = request.metric.upper()
        figure, axis = self._new_figure()
        plotted = 0
        errors: list[str] = []
        if request.ds == 0 and len(request.cases) == 1:
            case = request.cases[0]
            for ds in (1, 2, 3):
                try:
                    x, y = self._fragility_xy(case, metric, ds)
                    axis.plot(x, y * 100.0, linewidth=2.0, label=f"DS-{ds}")
                    plotted += 1
                except Exception as exc:
                    errors.append(f"DS-{ds}: {exc}")
        else:
            ds = max(1, request.ds)
            for case in request.cases:
                try:
                    x, y = self._fragility_xy(case, metric, ds)
                    axis.plot(x, y * 100.0, linewidth=2.0, label=case.display_name)
                    plotted += 1
                except Exception as exc:
                    errors.append(f"{case.display_name}: {exc}")
        self._require_plotted(plotted, errors)
        _set_axis_label(axis, "x", "Sa (g)")
        _set_axis_label(axis, "y", "Probability of exceedance (%)")
        axis.set_ylim(0, 100)
        self._finish(axis, request)
        return figure

    def _plot_fragility_surface(self, request: PlotRequest) -> Figure:
        metric = request.metric.upper()
        ds = max(1, request.ds)
        curves: list[tuple[float, np.ndarray, np.ndarray, str]] = []
        errors: list[str] = []
        for case in request.cases:
            try:
                if case.temperature is None:
                    raise ValueError("工况名称没有温度")
                x, y = self._fragility_xy(case, metric, ds)
                curves.append((float(case.temperature), x, y, case.display_name))
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        if len(curves) < 2:
            self._require_plotted(0, errors + ["易损性曲面至少需要两个不同温度工况。"])
        curves.sort(key=lambda item: item[0])
        x_min = max(float(np.min(item[1])) for item in curves)
        x_max = min(float(np.max(item[1])) for item in curves)
        x_common = np.linspace(x_min, x_max, 220)
        temperatures = np.asarray([item[0] for item in curves])
        z = np.vstack([np.interp(x_common, item[1], item[2]) for item in curves])
        x_grid, t_grid = np.meshgrid(x_common, temperatures)
        figure, axis = self._new_figure(projection="3d")
        surface = axis.plot_surface(x_grid, t_grid, z, cmap="viridis", linewidth=0, antialiased=True)
        colorbar = figure.colorbar(surface, ax=axis, shrink=0.68, pad=0.1)
        colorbar.set_label("Probability", fontsize=25, fontname="Times New Roman", labelpad=12)
        colorbar.ax.tick_params(direction="in", which="both", labelsize=18)
        _set_axis_label(axis, "x", "Sa (g)")
        _set_axis_label(axis, "y", f"Temperature ({CELSIUS})")
        z_label = "Exceedance probability"
        axis.set_zlabel(z_label, fontsize=25, fontname="Times New Roman", labelpad=12)
        axis.tick_params(axis="both", direction="in", which="both", labelsize=18)
        axis.zaxis.set_tick_params(direction="in", which="both", labelsize=18)
        if request.title.strip():
            title = normalize_temperature_label(request.title.strip())
            axis.set_title(title, fontsize=20, fontname=_label_font(title), pad=14)
        try:
            figure.tight_layout(rect=[0.01, 0.01, 0.93, 0.95])
        except (ValueError, TypeError):
            pass
        figure._paint_layout_right = 0.93
        figure._paint_layout_bottom = 0.01
        return figure

    def _plot_psdm(self, request: PlotRequest) -> Figure:
        metric = request.metric.upper()
        figure, axis = self._new_figure()
        plotted = 0
        errors: list[str] = []
        for case in request.cases:
            try:
                path = self._ida_workbook(case, "概率需求模型", metric)
                frame = pd.read_excel(path, sheet_name=0, header=None)
                x = pd.to_numeric(frame.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(frame.iloc[1:, 1], errors="coerce").to_numpy(dtype=float)
                fit_x = pd.to_numeric(frame.iloc[1:, 2], errors="coerce").to_numpy(dtype=float)
                fit_y = pd.to_numeric(frame.iloc[1:, 3], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                valid_fit = np.isfinite(fit_x) & np.isfinite(fit_y)
                axis.scatter(x[valid], y[valid], s=9, alpha=0.28)
                axis.plot(fit_x[valid_fit], fit_y[valid_fit], linewidth=2.0, label=case.display_name)
                plotted += 1
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        self._require_plotted(plotted, errors)
        _set_axis_label(axis, "x", "ln(IM)")
        _set_axis_label(axis, "y", f"ln({metric})")
        self._finish(axis, request)
        return figure

    def _plot_exceedance(self, request: PlotRequest) -> Figure:
        metric = request.metric.upper()
        ds = max(1, request.ds)
        figure, axis = self._new_figure()
        plotted = 0
        errors: list[str] = []
        for case in request.cases:
            try:
                path = self._ida_workbook(case, "易损性曲线", metric)
                book = pd.ExcelFile(path)
                sheet_index = min(ds, len(book.sheet_names) - 1)
                frame = pd.read_excel(path, sheet_name=sheet_index, header=None)
                actual_x = pd.to_numeric(frame.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)
                actual_y = pd.to_numeric(frame.iloc[1:, 1], errors="coerce").to_numpy(dtype=float)
                fit_x = pd.to_numeric(frame.iloc[1:, 2], errors="coerce").to_numpy(dtype=float)
                fit_y = pd.to_numeric(frame.iloc[1:, 3], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(actual_x) & np.isfinite(actual_y)
                valid_fit = np.isfinite(fit_x) & np.isfinite(fit_y)
                axis.scatter(actual_x[valid], actual_y[valid] * 100, s=9, alpha=0.18)
                axis.plot(fit_x[valid_fit], fit_y[valid_fit] * 100, linewidth=2.0, label=case.display_name)
                plotted += 1
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        self._require_plotted(plotted, errors)
        _set_axis_label(axis, "x", "Sa (g)")
        _set_axis_label(axis, "y", "Probability of exceedance (%)")
        axis.set_ylim(0, 100)
        self._finish(axis, request)
        return figure

    def _plot_convergence(self, request: PlotRequest) -> Figure:
        figure, axis = self._new_figure()
        width = 0.8 / max(1, len(request.cases))
        plotted = 0
        errors: list[str] = []
        all_gms: set[int] = set()
        case_counts: list[tuple[AnalysisCase, dict[int, int]]] = []
        for case in request.cases:
            if case.ida_dir is None:
                continue
            try:
                counts: dict[int, int] = {}
                for folder in case.ida_dir.iterdir():
                    if not folder.is_dir():
                        continue
                    match = re.match(r"(\d+)_\d+$", folder.name)
                    if match:
                        gm = int(match.group(1))
                        counts[gm] = counts.get(gm, 0) + 1
                if counts:
                    all_gms.update(counts)
                    case_counts.append((case, counts))
            except Exception as exc:
                errors.append(f"{case.display_name}: {exc}")
        gms = sorted(all_gms)
        for index, (case, counts) in enumerate(case_counts):
            positions = np.arange(len(gms)) + (index - (len(case_counts) - 1) / 2) * width
            axis.bar(positions, [counts.get(gm, 0) for gm in gms], width=width, label=case.display_name)
            plotted += 1
        self._require_plotted(plotted, errors)
        axis.set_xticks(np.arange(len(gms)), [str(gm) for gm in gms], rotation=90, fontsize=8)
        _set_axis_label(axis, "x", "Ground motion")
        _set_axis_label(axis, "y", "IDA analysis count")
        self._finish(axis, request)
        return figure
