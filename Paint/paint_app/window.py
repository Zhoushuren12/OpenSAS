"""PyQt5 main window for the click-based plotting application."""

from __future__ import annotations

from pathlib import Path
import traceback

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .catalog import AnalysisCase, ResultCatalog
from .plotting import PlotRequest, PlotService, normalize_temperature_label


ANALYSES = [("Pushover", "PO"), ("时程分析", "TH"), ("IDA 分析", "IDA")]

PLOT_TYPES = {
    "PO": [
        ("能力曲线：屋顶位移角—基底剪力", "capacity_drift"),
        ("能力曲线：屋顶位移—基底剪力", "capacity_displacement"),
        ("归一化能力曲线", "capacity_normalized"),
    ],
    "TH": [
        ("楼层响应剖面", "response_profile"),
        ("响应箱线图", "response_boxplot"),
        ("单条记录楼层峰值", "record_profile"),
        ("层间位移角时程", "time_history"),
        ("梁/柱铰楼层分布", "hinge_profile"),
    ],
    "IDA": [
        ("IDA 曲线（单条曲线 + 50% 分位线）", "ida_curves"),
        ("IDA 分位线（16% / 50% / 84%）", "ida_quantiles"),
        ("易损性曲线", "fragility"),
        ("温度—易损性曲面", "fragility_surface"),
        ("PSDM 概率需求模型", "psdm"),
        ("损伤超越概率", "exceedance"),
        ("IDA 计算次数检查", "convergence"),
    ],
}

TH_METRICS = [
    ("层间位移角 IDR", "IDR"),
    ("残余层间位移角 RIDR", "RIDR"),
    ("累积层间位移角 CIDR", "CIDR"),
    ("楼层加速度 PFA", "PFA"),
    ("楼层速度 PFV", "PFV"),
    ("楼层剪力", "SHEAR"),
    ("倒塌指标 DCF", "DCF"),
]

IDA_METRICS = [("层间位移角 IDR", "IDR"), ("残余层间位移角 RIDR", "RIDR"), ("楼层加速度 PFA", "PFA")]
HINGE_METRICS = [("梁铰", "BEAM"), ("柱铰", "COLUMN"), ("节点域", "PANEL")]
STATISTICS = [("中位数（50%）", "median"), ("平均值", "mean"), ("16% 分位", "p16"), ("84% 分位", "p84"), ("最大值", "max")]


class PaintMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("My-OpenSAS 绘图助手")
        self.resize(1450, 900)
        self.setMinimumSize(1120, 720)

        self.catalog: ResultCatalog | None = None
        self.cases_by_item: dict[int, AnalysisCase] = {}
        self.plot_service = PlotService()
        self.figure: Figure | None = None
        self.canvas: FigureCanvasQTAgg | None = None
        self.toolbar: NavigationToolbar2QT | None = None

        self._build_ui()
        self._connect_signals()
        self._apply_style()
        self._set_default_directory()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 10)
        root_layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("My-OpenSAS 绘图助手")
        title.setObjectName("title")
        subtitle = QLabel("选择分析结果，设置参数，点击生成")
        subtitle.setObjectName("subtitle")
        title_column = QVBoxLayout()
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        header.addLayout(title_column)
        header.addStretch()
        root_layout.addLayout(header)

        path_frame = QFrame()
        path_frame.setObjectName("pathFrame")
        path_layout = QHBoxLayout(path_frame)
        path_layout.setContentsMargins(12, 9, 12, 9)
        path_layout.addWidget(QLabel("结果目录"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("请选择 Output_data 文件夹")
        path_layout.addWidget(self.path_edit, 1)
        self.browse_button = QPushButton("浏览…")
        self.scan_button = QPushButton("扫描结果")
        self.scan_button.setObjectName("primarySmall")
        path_layout.addWidget(self.browse_button)
        path_layout.addWidget(self.scan_button)
        root_layout.addWidget(path_frame)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        controls = QWidget()
        controls.setMinimumWidth(365)
        controls.setMaximumWidth(455)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 8, 0)
        controls_layout.setSpacing(10)

        self.data_group = QGroupBox("1. 选择工况")
        data_layout = QVBoxLayout(self.data_group)
        self.case_list = QListWidget()
        self.case_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.case_list.setMinimumHeight(170)
        data_layout.addWidget(self.case_list)
        case_buttons = QHBoxLayout()
        self.select_all_button = QPushButton("全选可用工况")
        self.clear_button = QPushButton("清空")
        case_buttons.addWidget(self.select_all_button)
        case_buttons.addWidget(self.clear_button)
        data_layout.addLayout(case_buttons)
        controls_layout.addWidget(self.data_group)

        self.plot_group = QGroupBox("2. 选择图片")
        plot_form = QFormLayout(self.plot_group)
        self.analysis_combo = QComboBox()
        for label, value in ANALYSES:
            self.analysis_combo.addItem(label, value)
        self.plot_type_combo = QComboBox()
        plot_form.addRow("分析类型", self.analysis_combo)
        plot_form.addRow("图片类型", self.plot_type_combo)
        controls_layout.addWidget(self.plot_group)

        self.parameters_group = QGroupBox("3. 设置参数")
        self.parameter_form = QFormLayout(self.parameters_group)
        self.level_combo = QComboBox()
        self.metric_combo = QComboBox()
        self.statistic_combo = QComboBox()
        for label, value in STATISTICS:
            self.statistic_combo.addItem(label, value)
        self.record_combo = QComboBox()
        self.record_combo.setEditable(True)
        self.story_spin = QSpinBox()
        self.story_spin.setRange(1, 99)
        self.story_spin.setValue(1)
        self.ds_combo = QComboBox()
        self.ds_combo.addItem("全部损伤状态（单工况）", 0)
        for ds in range(1, 5):
            self.ds_combo.addItem(f"DS-{ds}", ds)
        self.sa_spin = QDoubleSpinBox()
        self.sa_spin.setRange(0.001, 20.0)
        self.sa_spin.setDecimals(3)
        self.sa_spin.setValue(1.0)
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("可留空")
        self.grid_check = QCheckBox("显示网格")
        self.grid_check.setChecked(True)

        self.parameter_rows: dict[str, tuple[QLabel, QWidget]] = {}
        for key, label, widget in [
            ("level", "地震水准", self.level_combo),
            ("metric", "响应指标", self.metric_combo),
            ("statistic", "统计方式", self.statistic_combo),
            ("record", "地震动记录", self.record_combo),
            ("story", "楼层", self.story_spin),
            ("ds", "损伤状态", self.ds_combo),
            ("sa", "Sa (g)", self.sa_spin),
            ("title", "图片标题", self.title_edit),
            ("grid", "辅助选项", self.grid_check),
        ]:
            row_label = QLabel(label)
            self.parameter_form.addRow(row_label, widget)
            self.parameter_rows[key] = (row_label, widget)
        controls_layout.addWidget(self.parameters_group)

        self.generate_button = QPushButton("生成图片")
        self.generate_button.setObjectName("primary")
        self.generate_button.setMinimumHeight(46)
        controls_layout.addWidget(self.generate_button)

        self.next_button = QPushButton("下一步：编辑图片 →")
        self.next_button.setMinimumHeight(40)
        self.next_button.setEnabled(False)
        controls_layout.addWidget(self.next_button)

        self.editor_group = QGroupBox("4. 编辑图片")
        editor_layout = QVBoxLayout(self.editor_group)
        editor_hint = QLabel("调整后点击“应用调整”，右侧预览会立即更新。")
        editor_hint.setWordWrap(True)
        editor_hint.setObjectName("editorHint")
        editor_layout.addWidget(editor_hint)
        editor_form = QFormLayout()

        self.edit_title = QLineEdit()
        self.edit_title.setPlaceholderText("留空则不显示标题")
        self.legend_visible = QCheckBox("显示图例")
        self.legend_visible.setChecked(True)
        self.legend_position = QComboBox()
        for label, value in [
            ("自动推荐", "auto"),
            ("图外右侧", "outside_right"),
            ("图外底部", "outside_bottom"),
            ("左上", "upper left"),
            ("右上", "upper right"),
            ("左下", "lower left"),
            ("右下", "lower right"),
            ("最佳位置", "best"),
        ]:
            self.legend_position.addItem(label, value)
        self.legend_columns = QSpinBox()
        self.legend_columns.setRange(1, 6)
        self.legend_columns.setValue(1)

        self.axis_font_size = QSpinBox()
        self.axis_font_size.setRange(10, 40)
        self.axis_font_size.setValue(25)
        self.tick_font_size = QSpinBox()
        self.tick_font_size.setRange(8, 32)
        self.tick_font_size.setValue(18)
        self.legend_font_size = QSpinBox()
        self.legend_font_size.setRange(8, 32)
        self.legend_font_size.setValue(15)
        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.3, 6.0)
        self.line_width.setSingleStep(0.1)
        self.line_width.setValue(2.0)
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(0.0, 16.0)
        self.marker_size.setSingleStep(0.5)
        self.marker_size.setValue(5.0)
        self.figure_width = QDoubleSpinBox()
        self.figure_width.setRange(4.0, 20.0)
        self.figure_width.setSingleStep(0.5)
        self.figure_width.setValue(10.0)
        self.figure_height = QDoubleSpinBox()
        self.figure_height.setRange(3.0, 16.0)
        self.figure_height.setSingleStep(0.5)
        self.figure_height.setValue(7.4)
        self.x_range_edit = QLineEdit()
        self.x_range_edit.setPlaceholderText("自动；或输入 0, 10")
        self.y_range_edit = QLineEdit()
        self.y_range_edit.setPlaceholderText("自动；或输入 0, 100")
        self.edit_grid = QCheckBox("显示主网格")
        self.edit_grid.setChecked(True)

        for label, widget in [
            ("标题", self.edit_title),
            ("图例", self.legend_visible),
            ("图例位置", self.legend_position),
            ("图例列数", self.legend_columns),
            ("坐标轴字号", self.axis_font_size),
            ("刻度字号", self.tick_font_size),
            ("图例字号", self.legend_font_size),
            ("线宽", self.line_width),
            ("标记大小", self.marker_size),
            ("画布宽度", self.figure_width),
            ("画布高度", self.figure_height),
            ("X 轴范围", self.x_range_edit),
            ("Y 轴范围", self.y_range_edit),
            ("网格", self.edit_grid),
        ]:
            editor_form.addRow(label, widget)
        editor_layout.addLayout(editor_form)

        self.apply_edit_button = QPushButton("应用调整")
        self.apply_edit_button.setObjectName("primary")
        self.reset_edit_button = QPushButton("恢复推荐样式")
        self.back_button = QPushButton("← 返回数据选择")
        editor_layout.addWidget(self.apply_edit_button)
        editor_actions = QHBoxLayout()
        editor_actions.addWidget(self.reset_edit_button)
        editor_actions.addWidget(self.back_button)
        editor_layout.addLayout(editor_actions)
        self.editor_group.setVisible(False)
        controls_layout.addWidget(self.editor_group)
        controls_layout.addStretch()
        splitter.addWidget(controls)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(8, 0, 0, 0)
        preview_header = QHBoxLayout()
        preview_title = QLabel("图片预览")
        preview_title.setObjectName("sectionTitle")
        preview_header.addWidget(preview_title)
        preview_header.addStretch()
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSuffix(" DPI")
        self.save_button = QPushButton("保存图片…")
        self.save_button.setEnabled(False)
        preview_header.addWidget(self.dpi_spin)
        preview_header.addWidget(self.save_button)
        preview_layout.addLayout(preview_header)

        self.canvas_layout = QVBoxLayout()
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addLayout(self.canvas_layout, 1)
        splitter.addWidget(preview)
        splitter.setSizes([405, 1000])

        self.status_label = QLabel("请选择包含分析结果的 Output_data 目录。")
        self.status_label.setObjectName("status")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root_layout.addWidget(self.status_label)

        empty = Figure(figsize=(8, 6), dpi=100)
        axis = empty.add_subplot(111)
        axis.text(0.5, 0.54, "尚未生成图片", ha="center", va="center", fontsize=20, fontname="SimSun", color="#748094")
        axis.text(0.5, 0.46, "先扫描结果目录，再从左侧选择工况和图片类型", ha="center", va="center", fontsize=11, fontname="SimSun", color="#9AA3B2")
        axis.set_axis_off()
        self._set_figure(empty)

    def _connect_signals(self) -> None:
        self.browse_button.clicked.connect(self._browse)
        self.scan_button.clicked.connect(lambda: self.scan_results())
        self.path_edit.returnPressed.connect(self.scan_results)
        self.analysis_combo.currentIndexChanged.connect(self._analysis_changed)
        self.plot_type_combo.currentIndexChanged.connect(self._plot_type_changed)
        self.case_list.itemChanged.connect(self._case_changed)
        self.level_combo.currentIndexChanged.connect(self._refresh_records)
        self.select_all_button.clicked.connect(lambda: self._set_all_cases(True))
        self.clear_button.clicked.connect(lambda: self._set_all_cases(False))
        self.generate_button.clicked.connect(self.generate_plot)
        self.next_button.clicked.connect(self._show_editor)
        self.back_button.clicked.connect(self._show_setup)
        self.apply_edit_button.clicked.connect(self.apply_figure_edits)
        self.reset_edit_button.clicked.connect(self.reset_figure_edits)
        self.save_button.clicked.connect(self.save_figure)

    def _set_default_directory(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.path_edit.setText(str(project_root / "Output_data"))
        self._analysis_changed()
        self.scan_results(show_empty_dialog=False)

    def _browse(self) -> None:
        start = self.path_edit.text().strip() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(self, "选择 Output_data 文件夹", start)
        if selected:
            self.path_edit.setText(selected)
            self.scan_results()

    def scan_results(self, *, show_empty_dialog: bool = True) -> None:
        try:
            catalog = ResultCatalog(self.path_edit.text().strip())
            cases = catalog.scan()
        except Exception as exc:
            self._show_error("无法扫描结果目录", exc)
            return
        self.catalog = catalog
        self.case_list.blockSignals(True)
        self.case_list.clear()
        self.cases_by_item.clear()
        for case in cases:
            item = QListWidgetItem(case.display_name)
            item.setData(Qt.UserRole, str(case.path))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.case_list.addItem(item)
            self.cases_by_item[id(item)] = case
        self.case_list.blockSignals(False)
        self._update_case_availability()
        self._refresh_levels()
        self.status_label.setText(catalog.summary() if cases else "该目录下没有识别到 Pushover、时程或 IDA 结果。")
        if not cases and show_empty_dialog:
            QMessageBox.information(
                self,
                "未发现结果",
                "没有识别到分析结果。\n\n请选择包含 MC8_<模型>_<温度> 工况文件夹的 Output_data 目录。",
            )

    def _analysis_changed(self, *_args) -> None:
        analysis = self.analysis_combo.currentData() or "PO"
        self.plot_type_combo.blockSignals(True)
        self.plot_type_combo.clear()
        for label, value in PLOT_TYPES[analysis]:
            self.plot_type_combo.addItem(label, value)
        self.plot_type_combo.blockSignals(False)
        self._update_case_availability()
        self._refresh_levels()
        self._plot_type_changed()

    def _update_case_availability(self) -> None:
        analysis = self.analysis_combo.currentData() or "PO"
        self.case_list.blockSignals(True)
        for row in range(self.case_list.count()):
            item = self.case_list.item(row)
            case = self.cases_by_item.get(id(item))
            enabled = case is not None and case.supports(analysis)
            flags = item.flags() | Qt.ItemIsUserCheckable
            item.setFlags(flags | Qt.ItemIsEnabled if enabled else flags & ~Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
            if case is not None:
                status = "、".join(case.available_analyses)
                item.setToolTip(f"{case.path}\n可用结果：{status}")
        self.case_list.blockSignals(False)

    def _set_all_cases(self, checked: bool) -> None:
        self.case_list.blockSignals(True)
        for row in range(self.case_list.count()):
            item = self.case_list.item(row)
            if item.flags() & Qt.ItemIsEnabled:
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.case_list.blockSignals(False)
        self._refresh_records()

    def _selected_cases(self) -> list[AnalysisCase]:
        selected: list[AnalysisCase] = []
        for row in range(self.case_list.count()):
            item = self.case_list.item(row)
            if item.checkState() == Qt.Checked and item.flags() & Qt.ItemIsEnabled:
                case = self.cases_by_item.get(id(item))
                if case is not None:
                    selected.append(case)
        return selected

    def _case_changed(self, _item: QListWidgetItem) -> None:
        self._refresh_levels()
        self._refresh_records()

    def _refresh_levels(self) -> None:
        if self.catalog is None:
            return
        current = self.level_combo.currentData()
        levels = self.catalog.levels(self._selected_cases() or self.catalog.cases)
        self.level_combo.blockSignals(True)
        self.level_combo.clear()
        for level in levels:
            self.level_combo.addItem(level, level)
        if current in levels:
            self.level_combo.setCurrentIndex(levels.index(current))
        elif "MCE" in levels:
            self.level_combo.setCurrentIndex(levels.index("MCE"))
        self.level_combo.blockSignals(False)
        self._refresh_records()

    def _refresh_records(self, *_args) -> None:
        analysis = self.analysis_combo.currentData() or "PO"
        plot_type = self.plot_type_combo.currentData()
        if analysis not in {"TH", "IDA"}:
            return
        current = self.record_combo.currentText().strip()
        level = self.level_combo.currentData() or ""
        raw = plot_type == "time_history"
        records: set[str] = set()
        for case in self._selected_cases():
            records.update(case.records(analysis, level, raw=raw))
        ordered = sorted(records, key=self._natural_key)
        self.record_combo.blockSignals(True)
        self.record_combo.clear()
        self.record_combo.addItems(ordered)
        if current in ordered:
            self.record_combo.setCurrentText(current)
        self.record_combo.blockSignals(False)

    @staticmethod
    def _natural_key(value: str) -> tuple:
        import re

        return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))

    def _plot_type_changed(self, *_args) -> None:
        analysis = self.analysis_combo.currentData() or "PO"
        plot_type = self.plot_type_combo.currentData()

        if plot_type == "hinge_profile":
            metrics = HINGE_METRICS
        elif analysis == "TH":
            metrics = TH_METRICS
        elif analysis == "IDA":
            metrics = IDA_METRICS
        else:
            metrics = []
        current_metric = self.metric_combo.currentData()
        self.metric_combo.clear()
        for label, value in metrics:
            self.metric_combo.addItem(label, value)
        values = [value for _, value in metrics]
        if current_metric in values:
            self.metric_combo.setCurrentIndex(values.index(current_metric))

        visible = {
            "level": analysis == "TH",
            "metric": bool(metrics) and plot_type != "time_history",
            "statistic": plot_type in {"response_profile", "hinge_profile"},
            "record": plot_type in {"record_profile", "time_history"},
            "story": plot_type == "time_history",
            "ds": plot_type in {"fragility", "fragility_surface", "exceedance"},
            "sa": False,
            "title": True,
            "grid": plot_type != "fragility_surface",
        }
        for key, (label, widget) in self.parameter_rows.items():
            label.setVisible(visible[key])
            widget.setVisible(visible[key])
        self._refresh_records()

    def generate_plot(self) -> None:
        cases = self._selected_cases()
        if not cases:
            QMessageBox.information(self, "请选择工况", "请至少勾选一个可用工况。")
            return
        request = PlotRequest(
            analysis=self.analysis_combo.currentData(),
            plot_type=self.plot_type_combo.currentData(),
            cases=tuple(cases),
            level=self.level_combo.currentData() or "",
            metric=self.metric_combo.currentData() or "IDR",
            statistic=self.statistic_combo.currentData() or "median",
            record=self.record_combo.currentText().strip(),
            story=self.story_spin.value(),
            ds=int(self.ds_combo.currentData() or 0),
            title=self.title_edit.text(),
            grid=self.grid_check.isChecked(),
            sa_value=self.sa_spin.value(),
        )
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.generate_button.setEnabled(False)
        self.status_label.setText("正在读取数据并生成图片…")
        QApplication.processEvents()
        try:
            figure = self.plot_service.create(request)
            self._set_figure(figure)
            self.save_button.setEnabled(True)
            self.next_button.setEnabled(True)
            self.status_label.setText(f"图片已生成：{self.plot_type_combo.currentText()}；共使用 {len(cases)} 个工况。")
        except Exception as exc:
            self._show_error("生成图片失败", exc)
        finally:
            self.generate_button.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def _set_figure(self, figure: Figure) -> None:
        if self.toolbar is not None:
            self.canvas_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
        if self.canvas is not None:
            self.canvas_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        self.figure = figure
        self.canvas = FigureCanvasQTAgg(figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setIconSize(self.toolbar.iconSize())
        self.canvas_layout.addWidget(self.toolbar)
        self.canvas_layout.addWidget(self.canvas, 1)
        self.canvas.draw_idle()

    def _show_editor(self) -> None:
        if self.figure is None or not self.save_button.isEnabled():
            QMessageBox.information(self, "请先生成图片", "请先完成第一步并生成基础图片。")
            return
        self.data_group.setVisible(False)
        self.plot_group.setVisible(False)
        self.parameters_group.setVisible(False)
        self.generate_button.setVisible(False)
        self.next_button.setVisible(False)
        self.editor_group.setVisible(True)
        axis = self.figure.axes[0] if self.figure.axes else None
        if axis is not None:
            self.edit_title.setText(axis.get_title())
            width, height = self.figure.get_size_inches()
            self.figure_width.setValue(float(width))
            self.figure_height.setValue(float(height))
            self.legend_visible.setChecked(axis.get_legend() is not None)
        self.status_label.setText("编辑模式：调整参数后点击“应用调整”。")

    def _show_setup(self) -> None:
        self.editor_group.setVisible(False)
        self.data_group.setVisible(True)
        self.plot_group.setVisible(True)
        self.parameters_group.setVisible(True)
        self.generate_button.setVisible(True)
        self.next_button.setVisible(True)
        self.status_label.setText("已返回数据选择；重新生成会替换当前预览。")

    @staticmethod
    def _parse_axis_range(text: str) -> tuple[float, float] | None:
        value = text.strip().replace("，", ",")
        if not value:
            return None
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError("坐标范围应输入两个数字，例如：0, 10")
        lower, upper = float(parts[0]), float(parts[1])
        if lower >= upper:
            raise ValueError("坐标范围的最小值必须小于最大值。")
        return lower, upper

    def apply_figure_edits(self) -> None:
        if self.figure is None or not self.figure.axes:
            return
        try:
            x_range = self._parse_axis_range(self.x_range_edit.text())
            y_range = self._parse_axis_range(self.y_range_edit.text())
            axis = self.figure.axes[0]
            self.figure.set_size_inches(self.figure_width.value(), self.figure_height.value(), forward=True)

            title = normalize_temperature_label(self.edit_title.text().strip())
            axis.set_title(title, fontsize=20, fontname="SimSun" if any("\u4e00" <= ch <= "\u9fff" for ch in title) else "Times New Roman", pad=14)
            axis.xaxis.label.set_fontsize(self.axis_font_size.value())
            axis.yaxis.label.set_fontsize(self.axis_font_size.value())
            if hasattr(axis, "zaxis"):
                axis.zaxis.label.set_fontsize(self.axis_font_size.value())
            axis.tick_params(axis="both", direction="in", which="major", labelsize=self.tick_font_size.value(), length=6, width=1.1)
            axis.tick_params(axis="both", direction="in", which="minor", length=3, width=0.8)

            for line in axis.get_lines():
                line.set_linewidth(self.line_width.value())
                if line.get_marker() not in {None, "", "None", "none"}:
                    line.set_markersize(self.marker_size.value())

            if x_range is None:
                axis.relim()
                axis.autoscale_view(scalex=True, scaley=False)
            else:
                axis.set_xlim(*x_range)
            if y_range is None:
                axis.relim()
                axis.autoscale_view(scalex=False, scaley=True)
            else:
                axis.set_ylim(*y_range)

            if not hasattr(axis, "zaxis"):
                axis.grid(self.edit_grid.isChecked(), which="major", linestyle="--", linewidth=0.6, alpha=0.20)

            old_legend = axis.get_legend()
            handles, labels = axis.get_legend_handles_labels()
            if old_legend is not None:
                old_legend.remove()
            layout_right = 0.95
            layout_bottom = 0.01
            if self.legend_visible.isChecked() and handles:
                position = self.legend_position.currentData()
                kwargs = {
                    "ncol": self.legend_columns.value(),
                    "frameon": False,
                    "fontsize": self.legend_font_size.value(),
                    "handlelength": 2.4,
                }
                if position == "outside_right":
                    kwargs.update(loc="center left", bbox_to_anchor=(1.01, 0.5))
                    layout_right = 0.78
                elif position == "outside_bottom":
                    kwargs.update(loc="upper center", bbox_to_anchor=(0.5, -0.14))
                    layout_bottom = 0.18
                elif position == "auto":
                    if len(handles) > 4:
                        kwargs.update(loc="center left", bbox_to_anchor=(1.01, 0.5))
                        layout_right = 0.78
                    else:
                        kwargs.update(loc="best")
                else:
                    kwargs.update(loc=position)
                axis.legend(handles, labels, **kwargs)

            try:
                self.figure.tight_layout(rect=[0.01, layout_bottom, layout_right, 0.95])
            except (ValueError, TypeError):
                pass
            self.figure._paint_layout_right = layout_right
            self.figure._paint_layout_bottom = layout_bottom
            self.canvas.draw_idle()
            self.status_label.setText("图片调整已应用，可继续修改或直接保存。")
        except Exception as exc:
            self._show_error("图片调整失败", exc)

    def reset_figure_edits(self) -> None:
        self.legend_position.setCurrentIndex(0)
        self.legend_columns.setValue(1)
        self.axis_font_size.setValue(25)
        self.tick_font_size.setValue(18)
        self.legend_font_size.setValue(15)
        self.line_width.setValue(2.0)
        self.marker_size.setValue(5.0)
        self.figure_width.setValue(10.0)
        self.figure_height.setValue(7.4)
        self.x_range_edit.clear()
        self.y_range_edit.clear()
        self.edit_grid.setChecked(True)
        self.legend_visible.setChecked(True)
        self.apply_figure_edits()

    def save_figure(self) -> None:
        if self.figure is None:
            return
        root = Path(self.path_edit.text().strip()).parent / "Paint" / "Figures"
        default_name = f"{self.analysis_combo.currentData()}_{self.plot_type_combo.currentData()}.png"
        default_path = root / default_name
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图片",
            str(default_path),
            "PNG 图片 (*.png);;PDF 矢量图 (*.pdf);;SVG 矢量图 (*.svg);;TIFF 图片 (*.tif *.tiff)",
        )
        if not path:
            return
        try:
            output = self.plot_service.export(self.figure, path, self.dpi_spin.value())
            self.status_label.setText(f"图片已保存：{output}")
        except Exception as exc:
            self._show_error("保存图片失败", exc)

    def _show_error(self, title: str, exc: Exception) -> None:
        self.status_label.setText(f"{title}：{exc}")
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        box = QMessageBox(QMessageBox.Critical, title, str(exc), parent=self)
        box.setDetailedText(detail)
        box.exec_()

    def _apply_style(self) -> None:
        self.setFont(QFont("Microsoft YaHei", 10))
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #F4F6F9; color: #273142; }
            QLabel#title { font-size: 24px; font-weight: 700; color: #18243A; }
            QLabel#subtitle { color: #738095; font-size: 11px; }
            QLabel#sectionTitle { font-size: 16px; font-weight: 650; color: #18243A; }
            QLabel#status { background: #EAF0F8; color: #44536A; border-radius: 5px; padding: 7px 10px; }
            QFrame#pathFrame { background: white; border: 1px solid #DCE2EB; border-radius: 7px; }
            QGroupBox { background: white; border: 1px solid #DCE2EB; border-radius: 7px; margin-top: 10px; padding-top: 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 11px; padding: 0 5px; color: #35445D; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {
                background: white; border: 1px solid #CBD3DF; border-radius: 5px; padding: 6px; min-height: 20px;
            }
            QLineEdit:focus, QComboBox:focus, QListWidget:focus { border: 1px solid #2E6FDB; }
            QPushButton { background: #FFFFFF; border: 1px solid #C7D0DE; border-radius: 5px; padding: 7px 12px; }
            QPushButton:hover { background: #EDF3FC; border-color: #8EA9D2; }
            QPushButton:disabled { color: #9CA6B5; background: #ECEFF3; }
            QPushButton#primary, QPushButton#primarySmall { background: #2868D7; color: white; border: none; font-weight: 650; }
            QPushButton#primary:hover, QPushButton#primarySmall:hover { background: #1E58BC; }
            QSplitter::handle { background: #E1E6ED; width: 1px; }
            QToolTip { background: #253247; color: white; border: none; padding: 5px; }
            """
        )
