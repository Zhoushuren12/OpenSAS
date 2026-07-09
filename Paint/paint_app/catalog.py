"""Fast, shallow discovery of My-OpenSAS analysis result folders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


_TEMPERATURE_SUFFIX = re.compile(r"^(?P<model>.+)_(?P<temperature>-?\d+(?:\.\d+)?)$")
_TH_OUT = re.compile(r".*_TH_(?P<level>.+?)_data_out$", re.IGNORECASE)
_TH_RAW = re.compile(r".*_TH_(?P<level>.+?)_data$", re.IGNORECASE)


def _temperature_sort_value(value: str | None) -> tuple[int, float | str]:
    if value is None:
        return (0, "")
    try:
        return (1, float(value))
    except ValueError:
        return (2, value)


@dataclass(frozen=True)
class AnalysisCase:
    """One model/temperature result directory."""

    path: Path
    model: str
    temperature: str | None
    po_dir: Path | None = None
    th_dirs: dict[str, Path] = field(default_factory=dict)
    th_raw_dirs: dict[str, Path] = field(default_factory=dict)
    ida_dir: Path | None = None
    ida_raw_dir: Path | None = None
    fragility_dir: Path | None = None

    @property
    def display_name(self) -> str:
        if self.temperature is None:
            return self.model
        return f"{self.model}  {self.temperature}{chr(0x2103)}"

    @property
    def available_analyses(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.po_dir is not None:
            values.append("PO")
        if self.th_dirs:
            values.append("TH")
        if self.ida_dir is not None or self.fragility_dir is not None:
            values.append("IDA")
        return tuple(values)

    def supports(self, analysis: str) -> bool:
        return analysis.upper() in self.available_analyses

    def records(self, analysis: str, level: str | None = None, *, raw: bool = False) -> list[str]:
        analysis = analysis.upper()
        root: Path | None
        if analysis == "IDA":
            root = self.ida_raw_dir if raw else self.ida_dir
        elif analysis == "TH":
            mapping = self.th_raw_dirs if raw else self.th_dirs
            root = mapping.get(level or "")
        else:
            return []
        if root is None or not root.is_dir():
            return []
        names = [item.name for item in root.iterdir() if item.is_dir()]
        return sorted(names, key=_natural_key)


def _natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


class ResultCatalog:
    """Catalog created from an ``Output_data`` directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.cases: list[AnalysisCase] = []

    def scan(self) -> list[AnalysisCase]:
        if not self.root.exists():
            raise FileNotFoundError(f"结果目录不存在：{self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"请选择文件夹：{self.root}")

        candidates = [item for item in self.root.iterdir() if item.is_dir()]
        if self._looks_like_case(self.root):
            candidates.insert(0, self.root)

        discovered = [case for folder in candidates if (case := self._inspect_case(folder)) is not None]
        self.cases = sorted(
            discovered,
            key=lambda case: (case.model.lower(), _temperature_sort_value(case.temperature)),
        )
        return self.cases

    @staticmethod
    def _looks_like_case(folder: Path) -> bool:
        try:
            names = {item.name for item in folder.iterdir() if item.is_dir()}
        except OSError:
            return False
        return any(name.endswith(("_PO_out", "_IDA_data_out", "_IDA_data_frag")) for name in names) or any(
            _TH_OUT.match(name) for name in names
        )

    @staticmethod
    def _parse_case_name(name: str) -> tuple[str, str | None]:
        payload = name[4:] if name.upper().startswith("MC8_") else name
        match = _TEMPERATURE_SUFFIX.match(payload)
        if match:
            return match.group("model"), match.group("temperature")
        return payload, None

    def _inspect_case(self, folder: Path) -> AnalysisCase | None:
        try:
            children = [item for item in folder.iterdir() if item.is_dir()]
        except OSError:
            return None

        po_dir: Path | None = None
        ida_dir: Path | None = None
        ida_raw_dir: Path | None = None
        fragility_dir: Path | None = None
        th_dirs: dict[str, Path] = {}
        th_raw_dirs: dict[str, Path] = {}

        for child in children:
            upper = child.name.upper()
            if upper.endswith("_PO_OUT"):
                po_dir = child
            elif upper.endswith("_IDA_DATA_OUT"):
                ida_dir = child
            elif upper.endswith("_IDA_DATA"):
                ida_raw_dir = child
            elif upper.endswith("_IDA_DATA_FRAG"):
                fragility_dir = child
            elif match := _TH_OUT.match(child.name):
                th_dirs[match.group("level").upper()] = child
            elif match := _TH_RAW.match(child.name):
                th_raw_dirs[match.group("level").upper()] = child

        if not any((po_dir, th_dirs, ida_dir, fragility_dir)):
            return None
        model, temperature = self._parse_case_name(folder.name)
        return AnalysisCase(
            path=folder,
            model=model,
            temperature=temperature,
            po_dir=po_dir,
            th_dirs=th_dirs,
            th_raw_dirs=th_raw_dirs,
            ida_dir=ida_dir,
            ida_raw_dir=ida_raw_dir,
            fragility_dir=fragility_dir,
        )

    def levels(self, cases: list[AnalysisCase] | None = None) -> list[str]:
        selected = cases if cases is not None else self.cases
        levels = {level for case in selected for level in case.th_dirs}
        preferred = {"CLE": 0, "DBE": 1, "MCE": 2}
        return sorted(levels, key=lambda value: (preferred.get(value, 99), value))

    def summary(self) -> str:
        counts = {
            "PO": sum(case.po_dir is not None for case in self.cases),
            "TH": sum(bool(case.th_dirs) for case in self.cases),
            "IDA": sum(case.ida_dir is not None or case.fragility_dir is not None for case in self.cases),
        }
        return (
            f"识别到 {len(self.cases)} 个工况；"
            f"Pushover {counts['PO']}，时程 {counts['TH']}，IDA {counts['IDA']}。"
        )
