from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

plt.rc("font", family="Times New Roman")


def read_ida_story_drift_profile(
    case_dir: Path,
    *,
    profile_file: str = "层间位移角.out",
    scale: float = 100.0,
    drop_base: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one IDA story-drift profile from:
    Output_data/MC8_<Model>_<Temp>/MC8_IDA_data_out/<record>/层间位移角.out
    """
    case_dir = Path(case_dir)
    f = case_dir / profile_file
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")

    data = np.asarray(np.loadtxt(f), dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError(f"Empty profile data: {f}")

    # Most outputs are [0, story1, story2, ...], remove base point if requested.
    if drop_base and data.size >= 2 and abs(float(data[0])) < 1e-12:
        data = data[1:]

    floors = np.arange(1, data.size + 1, dtype=int)
    idr = data * float(scale)
    return floors, idr


def plot_ida_story_drift_profiles(
    case_dirs: Sequence[Path],
    labels: Optional[Sequence[str]] = None,
    *,
    profile_file: str = "层间位移角.out",
    scale: float = 100.0,
    limit: Optional[float] = None,
    title: Optional[str] = None,
    xlabel: str = "IDR (%)",
    ylabel: str = "Floor",
    figsize: tuple[float, float] = (8, 6),
    lw: float = 1.8,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    invert_yaxis: bool = False,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Plot floor-vs-IDR profile curves for IDA cases."""
    case_dirs = [Path(p) for p in case_dirs]
    if labels is None:
        labels = [p.name for p in case_dirs]
    if len(labels) != len(case_dirs):
        raise ValueError("labels length must match case_dirs length")

    fig, ax = plt.subplots(figsize=figsize)
    max_floor = 0
    valid_count = 0

    for case_dir, label in zip(case_dirs, labels):
        try:
            floors, idr = read_ida_story_drift_profile(
                case_dir,
                profile_file=profile_file,
                scale=scale,
            )
            ax.plot(idr, floors, marker="o", linewidth=lw, label=label)
            max_floor = max(max_floor, int(floors.max()))
            valid_count += 1
        except Exception as exc:
            print(f"[skip] {case_dir}: {exc}")

    if valid_count == 0:
        print("[abort] No valid IDA story drift profile found.")
        plt.close(fig)
        return

    if limit is not None:
        ax.axvline(float(limit), color="black", linestyle="--", alpha=0.6, label="Limit")

    if title:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", direction="in", which="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_yticks(range(1, max_floor + 1))
    ax.legend(fontsize=12, loc="best")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if invert_yaxis:
        ax.invert_yaxis()

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_ida_out_case_dirs(
    base_dir: Path,
    model: str,
    temperatures: Iterable[str],
    record: str,
) -> list[Path]:
    """Build IDA output-case directories under MC8_IDA_data_out."""
    return [
        Path(base_dir) / f"MC8_{model}_{temp}" / "MC8_IDA_data_out" / record
        for temp in temperatures
    ]


if __name__ == "__main__":
    base_dir = Path("Output_data")
    model = "PFSDF"
    temperatures = ["-20", "0", "20", "40"]

    # Example: plot one record at multiple temperatures.
    # Record format is "<ground_motion_id>_<intensity_step>", e.g. "2_1", "10_15".
    for i in range(1, 30):
        record = f"8_{i}"
        case_dirs = build_ida_out_case_dirs(base_dir, model, temperatures, record)
        labels = [f"{t} C" for t in temperatures]

        plot_ida_story_drift_profiles(
            case_dirs,
            labels=labels,
            profile_file="层间位移角.out",
            scale=100.0,  # ratio -> percent
            limit=None,
            title=f"IDA Story Drift Profile ({record})",
            xlim=(0, 10),
            ylim=None,
            invert_yaxis=False,
            show=True,
        )
