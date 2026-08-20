from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from Paint.paint_app.catalog import AnalysisCase, ResultCatalog
from Paint.paint_app.plotting import PlotRequest, PlotService, _load_time


class CatalogTests(unittest.TestCase):
    def test_raw_component_sources_are_discovered(self) -> None:
        with TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "MC8_PFSDF_0"
            th_record = case_dir / "MC8_TH_MCE_data" / "1"
            cp_dir = case_dir / "MC8_CP" / "Cyclic_pushover"
            po_dir = case_dir / "MC8_PO" / "Pushover"
            for folder in (th_record, cp_dir, po_dir):
                folder.mkdir(parents=True)
            np.savetxt(th_record / "BeamSpring2_1R.out", [[1.0, 0.01]])
            np.savetxt(cp_dir / "ColSpring1_1T.out", [[2.0, 0.02]])
            np.savetxt(po_dir / "SMA1_1.out", [[3.0, 0.03]])

            cases = ResultCatalog(temporary).scan()

            self.assertEqual(len(cases), 1)
            case = cases[0]
            self.assertEqual(case.available_component_sources, ("TH", "CP", "PO"))
            self.assertTrue(case.supports("COMPONENT"))
            self.assertEqual(
                case.component_files("BEAM", "TH", "MCE", "1"),
                ["BeamSpring2_1R.out"],
            )
            self.assertEqual(
                case.component_files("COLUMN", "CP"),
                ["ColSpring1_1T.out"],
            )
            self.assertEqual(case.component_files("BRACE", "PO"), ["SMA1_1.out"])


class TimeHistoryTests(unittest.TestCase):
    def test_time_reader_selects_time_column_and_removes_static_prefix(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "Time.out"
            static_time = np.arange(0.1, 1.1, 0.1)
            transient_time = np.arange(0.02, 0.1, 0.02)
            time = np.concatenate((static_time, transient_time))
            np.savetxt(path, np.column_stack((np.zeros(time.size), time)))

            values, offset = _load_time(path)

            self.assertEqual(offset, 10)
            np.testing.assert_allclose(values, transient_time)

    def test_time_history_uses_sdr_and_real_time_axis(self) -> None:
        with TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "MC8_PFSDF_0"
            raw_root = case_dir / "MC8_TH_MCE_data"
            record_dir = raw_root / "8"
            record_dir.mkdir(parents=True)
            static_time = np.arange(0.1, 1.1, 0.1)
            transient_time = np.arange(0.02, 0.1, 0.02)
            time = np.concatenate((static_time, transient_time))
            response = np.concatenate((np.zeros(10), [0.001, -0.002, 0.003, -0.004]))
            np.savetxt(record_dir / "Time.out", np.column_stack((time, np.zeros(time.size))))
            np.savetxt(record_dir / "SDR5.out", response)
            case = AnalysisCase(
                path=case_dir,
                model="PFSDF",
                temperature="0",
                th_raw_dirs={"MCE": raw_root},
            )

            figure = PlotService().create(
                PlotRequest(
                    analysis="TH",
                    plot_type="time_history",
                    cases=(case,),
                    level="MCE",
                    record="8",
                    story=5,
                )
            )
            line = figure.axes[0].lines[0]

            np.testing.assert_allclose(line.get_xdata(), transient_time)
            np.testing.assert_allclose(line.get_ydata(), response[10:] * 100.0)


class ComponentPlotTests(unittest.TestCase):
    def test_hinge_and_brace_units_are_converted(self) -> None:
        with TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "MC8_PFSDF_0"
            cp_dir = case_dir / "MC8_CP" / "Cyclic_pushover"
            cp_dir.mkdir(parents=True)
            np.savetxt(cp_dir / "BeamSpring2_1R.out", [[1.0e6, 0.01], [-2.0e6, -0.02]])
            np.savetxt(cp_dir / "SMA1_1.out", [[1.0e3, 2.0], [-2.0e3, -3.0]])
            case = AnalysisCase(
                path=case_dir,
                model="PFSDF",
                temperature="0",
                cp_raw_dir=cp_dir,
            )
            service = PlotService()

            hinge = service.create(
                PlotRequest(
                    analysis="COMPONENT",
                    plot_type="beam_hysteresis",
                    cases=(case,),
                    source="CP",
                    component_file="BeamSpring2_1R.out",
                )
            )
            brace = service.create(
                PlotRequest(
                    analysis="COMPONENT",
                    plot_type="brace_hysteresis",
                    cases=(case,),
                    source="CP",
                    component_file="SMA1_1.out",
                )
            )

            np.testing.assert_allclose(hinge.axes[0].lines[0].get_xdata(), [0.01, -0.02])
            np.testing.assert_allclose(hinge.axes[0].lines[0].get_ydata(), [1.0, -2.0])
            np.testing.assert_allclose(brace.axes[0].lines[0].get_xdata(), [2.0, -3.0])
            np.testing.assert_allclose(brace.axes[0].lines[0].get_ydata(), [1.0, -2.0])


class PushoverPlotTests(unittest.TestCase):
    def test_lateral_force_pattern_is_normalized_and_numbered_by_floor(self) -> None:
        with TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "MC8_PFSDF_0"
            po_dir = case_dir / "MC8_PO" / "Pushover"
            po_dir.mkdir(parents=True)
            np.savetxt(po_dir / "Pattern.out", [1.0, 2.0, 3.0, 4.0])
            case = AnalysisCase(
                path=case_dir,
                model="PFSDF",
                temperature="0",
                po_raw_dir=po_dir,
            )

            figure = PlotService().create(
                PlotRequest(
                    analysis="PO",
                    plot_type="lateral_force_pattern",
                    cases=(case,),
                )
            )
            line = figure.axes[0].lines[0]

            np.testing.assert_allclose(line.get_xdata(), [0.1, 0.2, 0.3, 0.4])
            np.testing.assert_allclose(line.get_ydata(), [1.0, 2.0, 3.0, 4.0])
            self.assertAlmostEqual(float(np.sum(line.get_xdata())), 1.0)


if __name__ == "__main__":
    unittest.main()
