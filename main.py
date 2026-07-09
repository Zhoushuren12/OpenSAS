import sys
from pathlib import Path
sys.path.append(Path.cwd().as_posix())

from MRFcore.MRF import MRF
from MRFcore.DataProcessing import DataProcessing
from MRFcore.QuakeReadPushover import QuakeReadPushover
import numpy as np
from typing import Sequence, Optional


class FrameAnalysisRunner:
    """
    一键顺序运行：
      1) Pushover
      2) 读入/处理 Pushover 结果
      3) 多地震动时程分析（可配置 DBE/MCE/CLE）
      4) 读入/处理 时程结果

    通过减少重复代码，确保 SMABF / SMAFBF 两套工况一致可控。
    """

    def __init__(
        self,
        model_name: str,
        suffix: str = "_MCE_0.08_20",
        n_story: int = 8,
        n_bay: int = 3,
        heights: Optional[Sequence[int]] = None,
        spectrum_dir: Path = Path("Spectrum"),
        motion_ids: Sequence[str] = tuple(str(i) for i in range(1, 45)),
        display: bool = False
    ):
        self.model_name = model_name
        self.suffix = suffix
        self.n_story = n_story
        self.n_bay = n_bay
        self.heights = list(heights) if heights else [5500, 4300, 4300, 4300, 4300, 4300, 4300, 4300]
        self.spectrum_dir = spectrum_dir
        self.motion_ids = list(motion_ids)
        self.display = display

        # 统一的输出目录结构
        self.base_dir = Path("Output_data") / f"{self.model_name}{self.suffix or ''}"
        self.po_dir   = self.base_dir / "MC8_PO"                # pushover 输出
        # 时程输出示例：self.base_dir / f"MC8_TH_{level}_data"

    # ----------------- 内部工具 -----------------
    def _new_model(self, notes: str) -> MRF:
        return MRF(
            self.model_name,
            Nstory=self.n_story,
            Nbay=self.n_bay,
            heights=self.heights,
            notes=notes,
            script='py'
        )

    def _read_fundamental_period(self) -> float:
        """从 Pushover 输出中读取 T1（周期(s).out 第一行）。"""
        t1_path = self.base_dir / "MC8_PO_out" / "周期(s).out"
        if not t1_path.exists():
            raise FileNotFoundError(
                f"未找到基本周期文件：{t1_path}\n"
                f"请先运行 run_pushover() 或检查输出路径是否一致。"
            )
        T1 = float(np.loadtxt(t1_path)[0])
        return T1

    # ----------------- 公开步骤 -----------------
    def run_pushover(self, plot_result: bool = True, auto_quit: bool = True,):
        """
        运行 pushover 分析。
        若检测到已有结果 (周期(s).out) 且未设置 force=True，则跳过。
        """
        t1_path = self.base_dir / "MC8_PO_out" / "周期(s).out"

        # if t1_path.exists() and not force:
        #     print(f"[跳过] 已检测到现有 Pushover 结果: {t1_path}")
        #     return

        # 否则执行计算
        note = 'pushover analysis of an eight-story steel moment resisting frame'
        model = self._new_model(note)
        model.set_running_parameters(
            Output_dir=self.po_dir,
            display=plot_result,
            auto_quit=auto_quit,
            folder_exists='delete'
        )
        model.run_pushover(print_result=True)
        # QuakeReadPushover(self.po_dir)

        dp = DataProcessing(self.po_dir)
        dp.set_output_dir(self.po_dir.parent / (self.po_dir.name + '_out'), cover=2)
        dp.read_results('mode', 'IDR', 'CIDR', 'PFA', 'PFV', 'shear', 'panelZone',
                        'beamHinge', 'columnHinge', print_result=True)
        dp.read_pushover(H=35600, plot_result=False)

    def run_time_history(
        self,
        levels: Sequence[str] = ("DBE", "MCE", "ERE"),
        fv_duration: int = 30,
        parallel: int = 7,
        auto_quit: bool = True,
        save_SF: bool = True,
        plot_SF: bool = False
    ):
        # 读取 T1 以用于频带缩放
        T1 = self._read_fundamental_period()

        for level in levels:
            th_dir = self.base_dir / f"MC8_TH_{level}_data"

            note = 'time history of an eight-story steel moment resisting frame'
            model = self._new_model(note)

            # 选取地震动
            model.select_ground_motions(self.motion_ids, suffix='.txt')

            # 频带缩放参数 & 目标谱路径（Path 拼接避免反斜杠转义问题）
            para = (0.2 * T1, 1.5 * T1)
            spec_path = self.spectrum_dir / f"{level} Level Spectrum.txt"

            model.scale_ground_motions(
                method='c',
                para=para,
                path_spec_code=spec_path,
                save_SF=save_SF,
                plot=plot_SF
            )

            # 运行时程
            model.set_running_parameters(
                Output_dir=th_dir,
                fv_duration=fv_duration,
                display=self.display,
                auto_quit=auto_quit,
                folder_exists='overwrite'
            )
            model.run_time_history(print_result=False, parallel=parallel)

            # 读入/处理时程结果
            dp = DataProcessing(th_dir)
            dp.set_output_dir(th_dir.parent / (th_dir.name + '_out'), cover=1)
            dp.read_results('mode', 'IDR', 'CIDR', 'PFA', 'PFV', 'shear', 'panelZone',
                            'beamHinge', 'columnHinge', print_result=True)
            dp.read_th()

    def run_all(
        self,
        levels: Sequence[str] = ("DBE", "MCE", "ERE"),
        po_plot: bool = False
    ):
        """顺序：Pushover → 时程（多级）"""
        print(f"[{self.model_name}] 开始 Pushover...")
        self.run_pushover(plot_result=po_plot)
        print(f"[{self.model_name}] Pushover 完成，开始时程...")
        self.run_time_history(levels=levels)
        print(f"[{self.model_name}] 全流程完成。")

# ================== 用 法 示 例 ==================
if __name__ == "__main__":

    # temperature = ['-20','0','20','40']
    temperature = ['0','10','20','30','40']
    for temp in temperature:

    # 例1：运行 SMABF
        smabf = FrameAnalysisRunner(
            model_name=f"MC8_SMABF_{temp}",
            suffix="",
            display=False
        )
        
        # 单独跑：
        # smabf.run_pushover(plot_result=False)
        # smabf.run_time_history(levels=("DBE", "MCE", "ERE"))
        # smabf.run_time_history(levels=("ERE"))

        # # 一键顺序跑：
        # smabf.run_all(levels=("DBE", "MCE", "ERE"))

        # 例2：运行 PFSDF
        smafbf = FrameAnalysisRunner(
            model_name=f"MC8_PFSDF_{temp}",
            suffix="",
            display=False
        )
        # 单独跑：
        # smafbf.run_pushover(plot_result=False)
        # smafbf.run_time_history(levels=("ERE"))
        smafbf.run_time_history(levels=("DBE", "MCE"))
        
        # 一键顺序跑：
        # smafbf.run_all(levels=("DBE", "MCE", "ERE"))
