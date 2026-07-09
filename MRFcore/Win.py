from __future__ import annotations
import multiprocessing.queues
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from .MRF import MRF
import os
import sys
import re
import time
import multiprocessing
import subprocess
import datetime
import traceback
import json
import signal
import threading
import numpy as np
from pathlib import Path
from typing import Literal
from importlib import import_module
from collections.abc import Callable
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog
from ui.Win_running import Ui_Win_running


"""
时程分析、增量动力分析、Pushover分析监控窗口
作者：列文琛
更新：2024-03-05
更新：2024-04-07，优化代码
"""

class MyWin(QDialog):

    def __init__(
            self, main: MRF,
            running_case: Literal['IDA', 'TH', 'PO', 'CP'],
            IDA_para: tuple,
            print_result: bool):
        """IDA分析

        Args:
            main (MRF): MRF类\n
            running_case (str): 运行工况，'IDA', 'th'或'pushover'\n
            IDA_para (tuple): IDA参数\n
            * `T0`: 一阶周期
            * `RSA`: 加速度谱
            * `Sa0`: 初始强度
            * `Sa_incr`: 强度增量
            * `tol`: 倒塌点收敛容差
            * `max_ana`: 每条地震动最大计算次数
            * `test`: 程序调试用
            * `intensity_measure`: 地震动缩放方法
                * 1 - Sa(T)
                * 2 - Sa,avg 
            * `T_range`: 当`intensity_measure`为2时，给定的周期范围\n
            print_result (bool): 是否打印结果\n
        """
        super().__init__()
        self.ui = Ui_Win_running()
        self.main = main
        self.running_case = running_case  # "th" 或 "IDA"
        self.warning = 0  # 是否有警告信息
        self.current_gm = None
        if IDA_para:
            self.T0, self.RSA, self.Sa0, self.Sa_incr, self.tol, self.max_ana, self.test, self.intensity_measure, self.T_range = IDA_para
        self.print_result = print_result  # 是否输出运行过程中的消息
        self.openseespy_paras = None
        self.ui.setupUi(self)
        self.init_ui()
        QTimer.singleShot(0, self.run)

    def init_ui(self):
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint)
        self.set_ui_progrsssBar(('初始化中...', 0))
        if self.running_case == 'TH':
            if self.main.parallel:
                self.ui.label_3.setText('正在运行：多进程时程分析')
            else:
                self.ui.label_3.setText('正在运行：时程分析')
            self.ui.label_19.setEnabled(False)
            self.ui.label_20.setEnabled(False)
            self.ui.label_9.setText('（仅IDA适用）')
            self.ui.label_9.setEnabled(False)
            self.add_log('运行工况：时程分析\n')
        elif self.running_case == 'IDA':
            if self.main.parallel:
                self.ui.label_3.setText('正在运行：多进程IDA')
            else:
                self.ui.label_3.setText('正在运行：IDA')
            self.ui.label_20.setText('')
            self.add_log('运行工况：IDA\n')
        elif self.running_case == 'PO':
            self.ui.label_3.setText('正在运行：Pushover')
            self.add_log('运行工况：Pushover分析\n')
        elif self.running_case == 'CP':
            self.ui.label_3.setText('正在运行：Cyclic pushover')
            self.add_log('运行工况：Cyclic pushover分析\n')
        else:
            raise ValueError('【Error】参数 running_case 错误')
        if self.main.script == 'tcl':
            self.add_log('脚本类型：tcl\n')
        else:
            self.add_log('脚本类型：openseespy\n')
        if self.main.parallel:
            self.add_log(f'并行计算核心数：{self.main.parallel}\n')
        self.add_log(f'总地震动数量：{self.main.GM_N}\n')
        self.add_log(f'输出文件夹：{self.main.Output_dir}\n')
        self.time_project_begin = time.time()
        time_project_begin_struct = time.strftime("%Y/%m/%d\n%H:%M:%S", time.localtime(self.time_project_begin))
        self.add_log('开始时间：' + time_project_begin_struct.replace('\n', ' ') + '\n')
        if self.running_case == 'IDA':
            self.add_log(f'初始强度：{self.Sa0}g\n')
            self.add_log(f'强度增量：{self.Sa_incr}g\n')
            if self.main.trace_collapse:
                self.add_log(f'倒塌点收敛强度容差：{self.tol}g\n')
            else:
                self.add_log(f'倒塌点收敛强度容差：不追踪倒塌点\n')
            dict_ = {1: 'Sa(T)', 2: 'Sa,avg'}
            self.add_log(f'地震动强度指标：{dict_[self.intensity_measure]}\n')
        self.add_log('\n')
        self.ui.label_2.setText(time_project_begin_struct)  # 项目开始时间
        self.model: str = self.main.model_name
        self.ui.label_5.setText(self.model)  # 模型名称
        self.fv_duration: float = self.main.fv_duration
        if self.running_case in ['IDA', 'TH']:
            self.ui.label_17.setText(f'{self.fv_duration}s')  # 自由振动时长
        elif self.running_case in ['PO', 'CP']:
            self.ui.label_17.setText('')
        self.ui.pushButton_2.clicked.connect(self.copy_current_script)
        self.thread_run = None
        self.ui.pushButton.clicked.connect(self.kill)
        self.ui.pushButton_3.clicked.connect(lambda x: self.quit(self.current_gm))

    def set_ui_gm_info(self, tuple_: tuple):
        gm_name, intensity, duration, dt, NPTS = tuple_
        self.ui.label_7.setText(gm_name)  # 当前地震动
        self.ui.label_9.setText(intensity)  # 地震动强度
        self.ui.label_13.setText(str(duration) + 's')  # 持时
        self.ui.label_11.setText(str(dt) + 's')  # 步长
        self.ui.label_15.setText(str(NPTS))  # 步数

    def set_ui_progrsssBar(self, tuple_: tuple):
        text, val = tuple_
        self.ui.label_18.setText(text)
        self.ui.progressBar.setValue(val)

    def finished(self, state):
        # state: 1-顺利完成计算，2-中断
        time_project_end = time.time()
        time_project_end_struct = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time_project_end))
        time_cost = time_project_end - self.time_project_begin
        self.add_log(f'\n分析完成：{time_project_end_struct}\n')
        self.add_log(f'总耗时：{int(time_cost)}s\n')
        with open(self.main.dir_log / f'【{self.running_case.upper()}】{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time_project_end))}.log', 'w') as f:
            text = self.ui.textBrowser.toPlainText()
            f.write(text)
        with open(self.main.Output_dir / f'{self.main.log_name}.log', 'w') as f:
            text = self.ui.textBrowser.toPlainText()
            f.write(text)
        if self.warning == 1:
            with open(self.main.dir_log / f'【warning】{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time_project_end))}.log', 'w') as f:
                text = self.ui.textBrowser_2.toPlainText()
                f.write(text)
            with open(self.main.Output_dir / '警告.log', 'w') as f:
                text = self.ui.textBrowser_2.toPlainText()
                f.write(text)
        if state == 1:
            self.ui.label_18.setText('计算完成！')
            self.ui.progressBar.setValue(100)
        self.ui.pushButton_3.setEnabled(True)
        self.main.logger.success('所有分析已完成')
        if self.main.auto_quit:
            self.accept()
            app = QApplication.instance()
            if app is not None:
                app.quit()

    def run(self):
        self.thread_run = WorkerThread(self.main, self)
        self.thread_run.signal_set_ui.connect(self.set_ui_gm_info)
        self.thread_run.signal_set_progressBar.connect(self.set_ui_progrsssBar)
        self.thread_run.signal_finished.connect(self.finished)
        self.thread_run.signal_add_log.connect(self.add_log)
        self.thread_run.signal_add_warning.connect(self.add_warinng)
        self.thread_run.signal_send_openseespy_paras.connect(self.get_openseespy_paras)
        self.thread_run.start()

    def copy_current_script(self):
        if self.main.parallel > 0:
            QMessageBox.warning(self, '警告', '多进程计算不支持查看运行代码！')
            return
        if self.main.script == 'tcl':
            if self.running_case == 'PO':
                with open(self.main.dir_temp / f'temp_running_{self.main.model_name}_Pushover.{self.main.script}', 'r') as f:
                    text = f.read()
            elif self.running_case == 'CP':
                with open(self.main.dir_temp / f'temp_running_{self.main.model_name}_Cyclic_pushover.{self.main.script}', 'r') as f:
                    text = f.read()
            else:
                with open(self.main.dir_temp / f'temp_running_{self.main.model_name}_{self.current_gm}.{self.main.script}', 'r') as f:
                    text = f.read()
        else:
            QMessageBox.warning(self, '警告', 'openseespy脚本暂不支持查看运行代码！')
            return
            paras = self.openseespy_paras
            added_text = '\n\n'
            added_text += 'if name == "__main__":\n'
            added_text += f'result = run_openseespy({paras})'

        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self.add_log(f'已复制代码\n')
        QMessageBox.information(self, '提示', '已复制至剪切板。')

    def kill(self):
        if self.running_case == 'TH' and self.main.parallel > 0:
            QMessageBox.warning(self, '警告', '多进程时程分析不支持中断！')
            return
        if self.thread_run:
            if QMessageBox.question(self, '警告', '是否中断计算？\n（在计算完该地震动后）') == QMessageBox.Yes:
                self.thread_run.is_kill = 1
                self.thread_run.stop_event.set()
                self.add_log('在计算完该地震动后将中断计算!\n')

    def add_log(self, text_add):
        text = self.ui.textBrowser.toPlainText()
        text += text_add
        self.ui.textBrowser.setText(text)
        self.ui.textBrowser.verticalScrollBar().setValue(self.ui.textBrowser.verticalScrollBar().maximum())

    def add_warinng(self, text_add):
        self.ui.tabWidget.setTabText(1, '【警告】')
        text = self.ui.textBrowser_2.toPlainText()
        text += text_add
        self.ui.textBrowser_2.setText(text)
        self.ui.textBrowser_2.verticalScrollBar().setValue(self.ui.textBrowser_2.verticalScrollBar().maximum())
        self.warning = 1

    def quit(self, current_gm):
        # os.remove(f'{self.main.cwd}/temp_running_{self.main.model_name}_{current_gm}.tcl')
        pass

    def get_openseespy_paras(self, paras: list):
        self.openseespy_paras = paras


class WorkerThread(QThread):
    """opensees求解子线程
    """

    signal_set_ui = pyqtSignal(tuple)  # 运行过程中显示dt，NPTS等
    signal_set_progressBar = pyqtSignal(tuple)  # 运行过程中设置进度条
    signal_finished = pyqtSignal(int)  # 运行完成
    signal_add_log = pyqtSignal(str)  # 增加日志内容
    signal_add_warning = pyqtSignal(str)  # 增加警告内容
    signal_send_openseespy_paras = pyqtSignal(list)  # openseespy调用参数


    def __init__(self, main: MRF, mainWin: MyWin):
        super().__init__()
        self.main = main
        self.mainWin = mainWin
        self.model_name = self.main.model_name
        self.OS_path: str = main.OS_path
        self.is_kill = 0
        self._mp_manager = None
        if self.mainWin.running_case in ('TH', 'IDA') and self.main.parallel > 0:
            self._mp_manager = multiprocessing.Manager()
            self.stop_event = self._mp_manager.Event()
        else:
            self.stop_event = threading.Event()


    def modify_script1(self, Output_dir: Path, gm_name: str, dt: str | float,
                   NPTS: int | float, duration: float | str,
                   fv_duration: float | str, SF: float | str, num: int=None):
        """采用正则表达式修改脚本文件(仅当采用tcl脚本时需要修改)

        Args:
            Output_dir (Path): 输出文件夹路径
            gm_name (str): 地震动名
            dt (str | float): 步长
            NPTS (int | float): 步数
            duration (float | str): 持时
            fv_duration (float | str): 自由振动时长
            SF (float | str): 缩放系数
            num (int, optional): 当前地震动的序号
        """

        with open(self.main.dir_model / f'{self.main.model_name}.tcl', 'r') as f:
            text = f.read()
        pattern = re.compile(r'(set maxRunTime )[0-9.]+(;  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(float(self.main.maxRunTime)) + r'\g<2>', text) 
        pattern = re.compile(r'(set analysis_type ")[THPOCP]+(";  # \$\$\$)')
        self.find_pattern(pattern, text)
        if self.mainWin.running_case in ['TH', 'IDA']:
            text = pattern.sub(r'\g<1>' + 'TH' + r'\g<2>', text)
        elif self.mainWin.running_case == 'PO':
            text = pattern.sub(r'\g<1>' + 'PO' + r'\g<2>', text)
        elif self.mainWin.running_case == 'CP':
            text = pattern.sub(r'\g<1>' + 'CP' + r'\g<2>', text)
        else:
            self.main.logger.warning('无法进行正则匹配 (set analysis_type)')
        pattern = re.compile(r'(set MainFolder ").+(";  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + Output_dir.absolute().as_posix() + r'\g<2>', text)
        pattern = re.compile(r'(set GMname ").+(";  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + gm_name + r'\g<2>', text)
        pattern = re.compile(r'(set SubFolder ").+(";  # \$\$\$)')
        self.find_pattern(pattern, text)
        if num:
            text = pattern.sub(r'\g<1>' + f'{gm_name}_{num}' + r'\g<2>', text)
        else:
            text = pattern.sub(r'\g<1>' + gm_name + r'\g<2>', text)
        pattern = re.compile(r'(set GMdt )[0-9.]+(;  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(dt) + r'\g<2>', text)
        pattern = re.compile(r'(set GMpoints )\d+(;  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(NPTS) + r'\g<2>', text)
        pattern = re.compile(r'(set GMduration )[0-9.]+(;  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(duration) + r'\g<2>', text)
        pattern = re.compile(r'(set FVduration )[0-9.]+(;  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(fv_duration) + r'\g<2>', text)
        pattern = re.compile(r'(set EqSF )[0-9.]+(;  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(SF) + r'\g<2>', text)
        pattern = re.compile(r'(set GMFile ").+(";  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + self.main.dir_gm.as_posix() + f'/$GMname{self.main.suffix}' + r'\g<2>', text)
        pattern = re.compile(r'(set subroutines ").+(";  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + self.main.dir_subroutines.as_posix() + r'\g<2>', text)
        pattern = re.compile(r'(set temp ").+(";  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + self.main.dir_temp.as_posix() + r'\g<2>', text)
        pattern = re.compile(r'set ShowAnimation [01];  # \$\$\$')
        self.find_pattern(pattern, text)
        if not self.main.display:
            text = pattern.sub(r'set ShowAnimation 0;  # $$$', text)
        else:
            text = pattern.sub(r'set ShowAnimation 1;  # $$$', text)
        pattern = re.compile(r'set MPCO [01];  # \$\$\$')
        self.find_pattern(pattern, text)
        if self.main.mpco:
            text = pattern.sub(r'set MPCO 1;  # $$$', text)
        else:
            text = pattern.sub(r'set MPCO 0;  # $$$', text)
        pattern = re.compile(r'(set maxRoofDrift )[01.]+(;  # \$\$\$)')
        self.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(self.main.maxRoofDrift) + r'\g<2>', text)
        if self.mainWin.running_case in ['IDA', 'TH']:
            pattern = re.compile(r'(set CollapseDrift )[0-9.]+(;  # \$\$\$)')
            self.find_pattern(pattern, text)
            text = pattern.sub(r'\g<1>' + str(self.main.collapse_limit) + r'\g<2>', text)
        elif self.mainWin.running_case == 'CP':
            pattern = re.compile(r'(set RDR_path \[list )[0-9. -]+(\];  # \$\$\$)')
            self.find_pattern(pattern, text)
            s = [str(i) for i in self.main.RDR_path]
            s = ' '.join(s)
            text = pattern.sub(r'\g<1>' + s + r'\g<2>', text)
        with open(self.main.dir_temp / f'temp_running_{self.main.model_name}_{gm_name}.tcl', 'w') as f:
            f.write(text)


    @staticmethod
    def modify_script(
        dir_model: Path, model_name: Path, maxRunTime: float, running_case: str,
        dir_gm: Path, dir_subroutines: Path, dir_temp: Path, suffix: str, display: bool, mpco: bool, maxRoofDrift: float,
        Output_dir: Path, gm_name: str, dt: str | float,
        NPTS: int | float, duration: float | str, fv_duration: float | str,
        SF: float | str, collapse_limit: float, num: int=None):
        """采用正则表达式修改脚本文件(仅当采用tcl脚本时需要修改)

        Args:
            Output_dir (Path): 输出文件夹路径
            gm_name (str): 地震动名
            dt (str | float): 步长
            NPTS (int | float): 步数
            duration (float | str): 持时
            fv_duration (float | str): 自由振动时长
            SF (float | str): 缩放系数
            num (int, optional): 当前地震动的序号
        """

        with open(dir_model / f'{model_name}.tcl', 'r') as f:
            text = f.read()
        pattern = re.compile(r'(set maxRunTime )[0-9.]+(;  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(float(maxRunTime)) + r'\g<2>', text) 
        pattern = re.compile(r'(set analysis_type ")[THPOCP]+(";  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        if running_case in ['TH', 'IDA']:
            text = pattern.sub(r'\g<1>' + 'TH' + r'\g<2>', text)
        elif running_case == 'PO':
            text = pattern.sub(r'\g<1>' + 'PO' + r'\g<2>', text)
        elif running_case == 'CP':
            text = pattern.sub(r'\g<1>' + 'CP' + r'\g<2>', text)
        else:
            raise ValueError('无法进行正则匹配 (set analysis_type)')
        pattern = re.compile(r'(set MainFolder ").+(";  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + Output_dir.absolute().as_posix() + r'\g<2>', text)
        pattern = re.compile(r'(set GMname ").+(";  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + gm_name + r'\g<2>', text)
        pattern = re.compile(r'(set SubFolder ").+(";  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        if num:
            text = pattern.sub(r'\g<1>' + f'{gm_name}_{num}' + r'\g<2>', text)
        else:
            text = pattern.sub(r'\g<1>' + gm_name + r'\g<2>', text)
        pattern = re.compile(r'(set GMdt )[0-9.]+(;  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(dt) + r'\g<2>', text)
        pattern = re.compile(r'(set GMpoints )\d+(;  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(NPTS) + r'\g<2>', text)
        pattern = re.compile(r'(set GMduration )[0-9.]+(;  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(duration) + r'\g<2>', text)
        pattern = re.compile(r'(set FVduration )[0-9.]+(;  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(fv_duration) + r'\g<2>', text)
        pattern = re.compile(r'(set EqSF )[0-9.]+(;  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(SF) + r'\g<2>', text)
        pattern = re.compile(r'(set GMFile ").+(";  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + dir_gm.as_posix() + f'/$GMname{suffix}' + r'\g<2>', text)
        pattern = re.compile(r'(set subroutines ").+(";  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + dir_subroutines.as_posix() + r'\g<2>', text)
        pattern = re.compile(r'(set temp ").+(";  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + dir_temp.as_posix() + r'\g<2>', text)
        pattern = re.compile(r'set ShowAnimation [01];  # \$\$\$')
        WorkerThread.find_pattern(pattern, text)
        if not display:
            text = pattern.sub(r'set ShowAnimation 0;  # $$$', text)
        else:
            text = pattern.sub(r'set ShowAnimation 1;  # $$$', text)
        pattern = re.compile(r'set MPCO [01];  # \$\$\$')
        WorkerThread.find_pattern(pattern, text)
        if mpco:
            text = pattern.sub(r'set MPCO 1;  # $$$', text)
        else:
            text = pattern.sub(r'set MPCO 0;  # $$$', text)
        pattern = re.compile(r'(set maxRoofDrift )[01.]+(;  # \$\$\$)')
        WorkerThread.find_pattern(pattern, text)
        text = pattern.sub(r'\g<1>' + str(maxRoofDrift) + r'\g<2>', text)
        if running_case in ['IDA', 'TH']:
            pattern = re.compile(r'(set CollapseDrift )[0-9.]+(;  # \$\$\$)')
            WorkerThread.find_pattern(pattern, text)
            text = pattern.sub(r'\g<1>' + str(collapse_limit) + r'\g<2>', text)
        with open(dir_temp / f'temp_running_{model_name}_{gm_name}.tcl', 'w') as f:
            f.write(text)


    @staticmethod
    def find_pattern(pattern: re.Pattern, text: str):
        """
        如果找不到匹配值，或找到多个匹配值，则报错，
        确保有且只有一种匹配模式
        """
        res = re.findall(pattern, text)
        if len(res) == 0:
            raise ValueError(f'无法匹配: {pattern}')
        if len(res) > 1:
            raise ValueError(f'找到{len(res)}种匹配模式: {pattern}')


    def run(self):
        if self.mainWin.running_case == 'TH' and not self.main.parallel:
            self.run_th()
        elif self.mainWin.running_case == 'TH' and self.main.parallel:
            self.run_th_parallel(self.main.parallel)
        elif self.mainWin.running_case == 'IDA' and not self.main.parallel:
            self.run_IDA()
        elif self.mainWin.running_case == 'IDA' and self.main.parallel:
            if self.main.script == 'py':
                self.run_IDA_parallel_py(self.main.parallel)
            else:
                self.run_IDA_parallel_tcl(self.main.parallel)
        elif self.mainWin.running_case == 'PO':
            self.run_pushover()
        elif self.mainWin.running_case == 'CP':
            self.run_cyclic_pushover()


    def get_queue(self, queue: multiprocessing.Queue):
        """进程通讯"""
        finished_GM = 0  # 已计算完成的地震动
        while True:
            if not queue.empty():
                message = queue.get()
                flag, text = message
                if flag == 'a':  # 地震动开始
                    self.signal_add_log.emit(text)
                elif flag == 'b':  # 地震动完成
                    self.signal_add_log.emit(text)
                    finished_GM += 1
                elif flag == 'c' or flag == 'd':  # 该次计算开始/完成
                    self.signal_add_log.emit(text)
                elif flag == 'e':  # 首次计算即倒塌
                    self.signal_add_warning.emit(text)
                    finished_GM += 1
                elif flag == 'f':  # 超过最大计算次数仍未找到倒塌点
                    self.signal_add_warning.emit(text)
                elif flag == 'g':  # 不收敛
                    self.signal_add_warning.emit(text)
                elif flag == 'h':  # 超过最大计算时间
                    self.signal_add_warning.emit(text)
                elif flag == 'i':  # 异常
                    error, tb = text
                    print(tb)
                    raise error
                self.signal_set_progressBar.emit((f'已完成地震动数量：{finished_GM}', int(finished_GM / self.main.GM_N * 100)))
                if finished_GM == self.main.GM_N:
                    break


    def run_th(self):
        for idx in range(self.main.GM_N):
            if self.is_kill == 1:
                self.signal_finished.emit(2)
                break
            self.main.logger.info(f'正在计算第{idx+1}条地震动 ({idx+1}/{self.main.GM_N})')
            gm_name: str = self.main.GM_names[idx]
            self.mainWin.current_gm = gm_name
            dt: float = self.main.GM_dts[idx]
            NPTS: int = self.main.GM_NPTS[idx]
            duration: float = self.main.GM_durations[idx]
            fv_duration: float = self.main.fv_duration
            SF: float = self.main.GM_SF[idx]
            self.signal_set_ui.emit((gm_name, '（仅IDA适用）', duration, dt, NPTS))
            self.signal_set_progressBar.emit((f'正在计算地震动：{idx+1}/{self.main.GM_N}', int(idx / self.main.GM_N * 100)))
            time_gm_start = time.time()
            self.signal_add_log.emit(f'开始：{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time_gm_start))}\n')
            self.signal_add_log.emit(f'地震动：{gm_name}\n')
            self.signal_add_log.emit(f'步长：{dt}s\n')
            self.signal_add_log.emit(f'步数：{NPTS}\n')
            self.signal_add_log.emit(f'持时：{duration}s\n')
            self.signal_add_log.emit(f'自由振动时间：{fv_duration}s\n')
            self.signal_add_log.emit(f'缩放系数：{SF:.3f}\n')
            if self.main.script == 'tcl':
                self.modify_script1(self.main.Output_dir, gm_name, dt, NPTS, duration, fv_duration, SF)
                cmd = f'"{self.OS_path}" "{self.main.dir_temp}/temp_running_{self.main.model_name}_{gm_name}.tcl"'
                # 运行分析
                if self.mainWin.print_result:
                    subprocess.call(cmd)
                else:
                    subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(self.main.dir_temp / f'{gm_name}_CollapseState.txt'):
                    collapsed = 1
                    self.signal_add_log.emit(f'倒塌：是\n')
                    os.remove(self.main.dir_temp / f'{gm_name}_CollapseState.txt')
                else:
                    collapsed = 0
                    self.signal_add_log.emit(f'倒塌：否\n')
                if not os.path.exists(self.main.Output_dir / gm_name):
                    os.makedirs(self.main.Output_dir / gm_name)
            else:
                module = import_module(f'models.{self.main.model_name}')
                run_openseespy = getattr(module, 'run_openseespy')
                maxRunTime = self.main.maxRunTime
                analysis_type = 'TH'
                if self.main.display:
                    ShowAnimation = True
                else:
                    ShowAnimation = False
                if self.main.mpco:
                    MPCO = True
                else:
                    MPCO = False
                MainFolder = self.main.Output_dir
                GMname = gm_name
                SubFolder = gm_name
                GMdt = dt
                GMpoints = NPTS
                GMduration = duration
                FVduration = fv_duration
                EqSF = SF
                GMFile = self.main.dir_gm / f'{gm_name}{self.main.suffix}'
                maxRoofDrift = 0.1
                collapse_limit = self.main.collapse_limit
                paras = [maxRunTime, analysis_type, ShowAnimation, MPCO, MainFolder, GMname, SubFolder,
                         GMdt, GMpoints, GMduration, FVduration, EqSF, GMFile, maxRoofDrift, collapse_limit, None]
                self.signal_send_openseespy_paras.emit(paras)
                result = run_openseespy(*paras)
                if result[2]:
                    collapsed = 1  # 分析完成，倒塌
                else:
                    collapsed = 0  # 分析完成，未倒塌
                if result[0] == 2:
                    s = '倒塌' if collapsed else '未倒塌'
                    self.signal_add_warning.emit(f'{gm_name}分析不收敛({s})')
                elif result[0] == 3:
                    self.signal_add_warning.emit(f'{gm_name}超过最大分析时间')
            Sa = None
            if self.main.method == 'd':
                Sa = self.main.th_para  # 指定PGA
            elif self.main.method == 'i':
                Sa = self.main.th_para[1]  # 指定Sa(Ta)
            elif self.main.method == 'j':
                Sa = self.main.th_para[2]  # 指定Sa,avg
            if Sa:
                np.savetxt(self.main.Output_dir / gm_name / 'Sa.dat', np.array([Sa]))
            with open(self.main.Output_dir / gm_name / 'isCollapsed.dat', 'w') as f:
                f.write(str(collapsed))
            if self.main.script == 'tcl':
                os.remove(self.main.dir_temp / f'temp_running_{self.main.model_name}_{self.mainWin.current_gm}.tcl')
            time_gm_end = time.time()
            time_cost = time_gm_end - time_gm_start
            self.signal_add_log.emit(f'结束：{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time_gm_end))}\n')
            self.signal_add_log.emit(f'该地震动计算耗时：{round(time_cost, 2)}s\n\n')
        else:
            self.signal_finished.emit(1)


    def run_th_parallel(self, processes: int):
        self.main.logger.info(f'正在进行多进程时程分析')
        self.signal_set_ui.emit(('', '', '', '', ''))
        queue = multiprocessing.Manager().Queue()  # 子进程向主进程通信
        script = self.main.script
        dir_model = self.main.dir_model
        dir_gm = self.main.dir_gm
        dir_subroutines = self.main.dir_subroutines
        dir_temp = self.main.dir_temp
        model_name = self.main.model_name
        mpco = self.main.mpco
        maxRunTime = self.main.maxRunTime
        Output_dir = self.main.Output_dir
        dir_gm = self.main.dir_gm
        suffix = self.main.suffix
        print_result = self.mainWin.print_result
        method = self.main.method
        th_para = self.main.th_para
        OS_path = self.main.OS_path
        collapse_limit = self.main.collapse_limit
        ls_paras: list[tuple] = []
        for idx in range(self.main.GM_N):
            gm_name: str = self.main.GM_names[idx]
            dt: float = self.main.GM_dts[idx]
            NPTS: int = self.main.GM_NPTS[idx]
            duration: float = self.main.GM_durations[idx]
            fv_duration: float = self.main.fv_duration
            SF = self.main.GM_SF[idx]
            paras = (queue, script, dir_model, dir_gm, dir_subroutines, dir_temp, model_name, gm_name, 
                     mpco, dt, NPTS, duration, fv_duration, maxRunTime, Output_dir, suffix,
                     SF, method, th_para, OS_path, print_result, collapse_limit)
            ls_paras.append(paras)
        pool = None
        try:
            pool = multiprocessing.Pool(processes, initializer=_ignore_keyboard_interrupt)
            results = []
            for i in range(self.main.GM_N):
                result = pool.apply_async(run_single_th, ls_paras[i])  # 设置进程池
                results.append(result)
            self.get_queue(queue)
            for result in results:
                output = result.get()
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            self.stop_event.set()
            if pool is not None:
                pool.terminate()
                pool.join()
            self.signal_add_warning.emit('Parallel analysis interrupted; unfinished tasks can be resumed when checkpoints are available.\n')
            self.signal_finished.emit(2)
            return
        except Exception:
            self.stop_event.set()
            if pool is not None:
                pool.terminate()
                pool.join()
            raise
        self.signal_finished.emit(1)
    


    def run_IDA(self):
        Sa0, Sa_incr, tol, max_ana = self.mainWin.Sa0, self.mainWin.Sa_incr, self.mainWin.tol, self.mainWin.max_ana
        checkpoint_name = getattr(self.main, 'IDA_checkpoint_name', '_ida_checkpoint.json')
        resume = getattr(self.main, 'IDA_resume', True)
        signature = _ida_signature(
            self.main.model_name, self.main.GM_names, self.mainWin.T0, Sa0, Sa_incr,
            tol, max_ana, self.mainWin.intensity_measure, self.mainWin.T_range,
            self.main.trace_collapse, self.main.collapse_limit
        )
        _ida_init_summary(self.main.Output_dir, checkpoint_name, signature)
        all_break = 0
        for idx in range(self.main.GM_N):
            if all_break == 1:
                self.signal_finished.emit(2)
                break
            self.main.logger.info(f'正在计算第{idx+1}条地震动 ({idx+1}/{self.main.GM_N})')
            gm_name: str = self.main.GM_names[idx]
            self.mainWin.current_gm = gm_name
            dt: float = self.main.GM_dts[idx]
            NPTS: int = self.main.GM_NPTS[idx]
            duration: float = self.main.GM_durations[idx]
            fv_duration: float = self.main.fv_duration
            T: np.ndarray = self.main.T
            RSA = self.mainWin.RSA[idx]
            Sa_current = Sa0
            if self.mainWin.intensity_measure == 1:
                Sa_original = self.Sa(T, RSA, self.mainWin.T0)  # 以Sa(T)作为地震动强度指标
            elif self.mainWin.intensity_measure == 2:
                Ta, Tb = self.mainWin.T_range
                Sa_range = RSA[(Ta <= T) & (T <= Tb)]
                Sa_avg = self.geometric_mean(Sa_range)  # 简单几何平均数
                Sa_original = Sa_avg  
            self.signal_set_progressBar.emit((f'正在计算地震动：{idx+1}/{self.main.GM_N}', int(idx / self.main.GM_N * 100)))
            state = _ida_resume_state(
                self.main.Output_dir, checkpoint_name, signature, gm_name,
                Sa0, Sa_incr, tol, max_ana, self.main.trace_collapse, resume
            )
            if state.get('status') == 'completed':
                self.signal_add_log.emit(f'{gm_name} checkpoint completed, skip\n')
                self.main.logger.success(f'第{idx+1}条地震动checkpoint已完成，跳过')
                continue
            Sa_current = float(state.get('next_Sa', Sa0))
            iter_state = int(state.get('iter_state', 0))
            Sa_l = float(state.get('Sa_l', 0))
            Sa_r = float(state.get('Sa_r', 100000))
            completed_points = list(state.get('completed_points', []))
            tasks = dict(state.get('tasks', {}))
            start_run_num = int(state.get('next_run_num', len(completed_points)))
            if start_run_num > 0:
                self.signal_add_log.emit(f'{gm_name} resume from checkpoint: completed {start_run_num}, next Sa={Sa_current}g\n')
            for run_num in range(start_run_num, max_ana):
                if self.is_kill == 1:
                    all_break = 1
                    break
                self.main.logger.info(f'\t第{idx+1}条地震动第{run_num+1}次计算')
                self.mainWin.ui.label_20.setText(str(run_num + 1))
                Sa_current = round(Sa_current, 5)
                SF = Sa_current / Sa_original
                task_key = _ida_task_key(run_num + 1)
                tasks[task_key] = {
                    'task_id': f'{gm_name}_{run_num + 1}',
                    'gm_name': gm_name,
                    'run_num': run_num + 1,
                    'folder': f'{gm_name}_{run_num + 1}',
                    'status': 'running',
                    'Sa': float(Sa_current),
                    'SF': float(SF),
                    'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
                }
                _ida_write_record(self.main.Output_dir, checkpoint_name, signature, gm_name, {
                    'status': 'running',
                    'reason': None,
                    'completed_points': completed_points,
                    'tasks': tasks,
                    'next_run_num': run_num,
                    'next_Sa': float(Sa_current),
                    'iter_state': iter_state,
                    'Sa_l': float(Sa_l),
                    'Sa_r': float(Sa_r),
                })
                self.signal_set_ui.emit((gm_name, f'{Sa_current}g', duration, dt, NPTS))
                time_gm_start = time.time()
                self.signal_add_log.emit(f'开始：{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time_gm_start))}\n')
                self.signal_add_log.emit(f'地震动：{gm_name}\n')
                self.signal_add_log.emit(f'步长：{dt}s\n')
                self.signal_add_log.emit(f'步数：{NPTS}\n')
                self.signal_add_log.emit(f'持时：{duration}s\n')
                self.signal_add_log.emit(f'自由振动时间：{fv_duration}s\n')
                self.signal_add_log.emit(f'缩放系数：{SF:.3f}\n')
                if self.mainWin.intensity_measure == 1:
                    text_Sa = 'Sa(T1)'
                elif self.mainWin.intensity_measure == 2:
                    text_Sa = 'Sa,avg'
                self.signal_add_log.emit(f'{text_Sa}：{Sa_current}g\n')
                result_status = 'ok'
                if self.main.script == 'tcl':
                    self.modify_script1(self.main.Output_dir, gm_name, dt, NPTS, duration, fv_duration, SF, run_num+1)
                    cmd = f'"{self.OS_path}" "{self.main.dir_temp}/temp_running_{self.main.model_name}_{gm_name}.tcl"'
                    if self.mainWin.test:
                        time.sleep(1)  # 模拟耗时工作
                    else:
                        # 运行分析
                        if self.mainWin.print_result:
                            subprocess.call(cmd)
                        else:
                            subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if os.path.exists(self.main.dir_temp/ f'{gm_name}_CollapseState.txt'):
                        collapsed = 1
                        self.signal_add_log.emit(f'倒塌：是\n')
                        os.remove(self.main.dir_temp / f'{gm_name}_CollapseState.txt')
                    else:
                        collapsed = 0
                        self.signal_add_log.emit(f'倒塌：否\n')
                else:
                    module = import_module(f'models.{self.main.model_name}')
                    run_openseespy = getattr(module, 'run_openseespy')
                    maxRunTime = self.main.maxRunTime
                    analysis_type = 'TH'
                    if self.main.display:
                        ShowAnimation = True
                    else:
                        ShowAnimation = False
                    if self.main.mpco:
                        MPCO = True
                    else:
                        MPCO = False
                    MainFolder = self.main.Output_dir
                    GMname = gm_name
                    SubFolder = f'{gm_name}_{run_num+1}'
                    GMdt = dt
                    GMpoints = NPTS
                    GMduration = duration
                    FVduration = fv_duration
                    EqSF = SF
                    GMFile = self.main.dir_gm / f'{gm_name}{self.main.suffix}'
                    maxRoofDrift = 0.1
                    collapse_limit = self.main.collapse_limit
                    print_result = self.mainWin.print_result
                    paras = [maxRunTime, analysis_type, ShowAnimation, MPCO, MainFolder, GMname, SubFolder,
                            GMdt, GMpoints, GMduration, FVduration, EqSF, GMFile, maxRoofDrift, collapse_limit, None]
                    self.signal_send_openseespy_paras.emit(paras)
                    with HiddenPrints(not print_result):
                        result = run_openseespy(*paras)
                    result_status = 'ok'
                    if result[2]:
                        collapsed = 1  # 分析完成，倒塌
                    else:
                        collapsed = 0  # 分析完成，未倒塌
                    if result[0] == 2:
                        s = '倒塌' if collapsed else '未倒塌'
                        self.signal_add_warning.emit(f'{gm_name}分析不收敛({s})')
                    elif result[0] == 3:
                        self.signal_add_warning.emit(f'{gm_name}超过最大分析时间')
                if not os.path.exists(self.main.Output_dir / f'{gm_name}_{run_num+1}'):
                    os.makedirs(self.main.Output_dir / f'{gm_name}_{run_num+1}')
                np.savetxt(self.main.Output_dir / f'{gm_name}_{run_num+1}/Sa.dat', np.array([Sa_current]))
                with open(self.main.Output_dir / f'{gm_name}_{run_num+1}/isCollapsed.dat', 'w') as f:
                    f.write(str(collapsed))
                time_gm_end = time.time()
                time_cost = time_gm_end - time_gm_start
                point = {
                    'run_num': run_num + 1,
                    'Sa': float(Sa_current),
                    'SF': float(SF),
                    'collapsed': bool(collapsed),
                    'elapsed_seconds': round(time_cost, 3),
                    'folder': f'{gm_name}_{run_num + 1}',
                    'result_status': result_status,
                }
                completed_points.append(point)
                tasks[task_key] = {
                    **tasks.get(task_key, {}),
                    'status': 'completed',
                    'collapsed': bool(collapsed),
                    'elapsed_seconds': round(time_cost, 3),
                    'result_status': result_status,
                    'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
                }
                self.signal_add_log.emit(f'结束：{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time_gm_end))}\n')
                self.signal_add_log.emit(f'该地震动已计算次数：{run_num+1}\n')
                self.signal_add_log.emit(f'该地震动计算耗时：{round(time_cost, 2)}s\n\n')
                if self.mainWin.test:
                    collapsed = 0
                if self.main.trace_collapse:
                    # 追踪倒塌
                    if run_num == 0 and collapsed == 1:
                        self.signal_add_warning.emit(f'{gm_name}首次计算即倒塌！\n')
                        self.main.logger.warning(f'{gm_name}首次计算即倒塌！')
                        _ida_write_record(self.main.Output_dir, checkpoint_name, signature, gm_name, {
                            'status': 'completed',
                            'reason': 'first_run_collapse',
                            'completed_points': completed_points,
                            'tasks': tasks,
                            'next_run_num': run_num + 1,
                            'next_Sa': float(Sa_current),
                            'iter_state': iter_state,
                            'Sa_l': float(Sa_l),
                            'Sa_r': float(Sa_current),
                        })
                        break
                    if collapsed == 0 and iter_state == 0:
                        # 如果未倒塌，且不处于迭代状态
                        Sa_l = Sa_current
                        Sa_current += Sa_incr
                    else:
                        # 在迭代状态下，即已经出现过倒塌后
                        iter_state = 1  # 进入迭代状态
                        if collapsed == 1:
                            # 若迭代状态下倒塌，更新最小倒塌强度
                            Sa_r = min(Sa_current, Sa_r)
                        else:
                            # 若迭代状态下未倒塌，更新最大未倒塌强度
                            Sa_l = max(Sa_current, Sa_l)
                        if Sa_l > Sa_r:
                            raise ValueError('最小倒塌强度大于最大未倒塌强度!')
                        if Sa_r - Sa_l < tol:
                            _ida_write_record(self.main.Output_dir, checkpoint_name, signature, gm_name, {
                                'status': 'completed',
                                'reason': 'collapse_tolerance_reached',
                                'completed_points': completed_points,
                                'tasks': tasks,
                                'next_run_num': run_num + 1,
                                'next_Sa': float(Sa_current),
                                'iter_state': iter_state,
                                'Sa_l': float(Sa_l),
                                'Sa_r': float(Sa_r),
                            })
                            break  # 满足收敛容差，完成当前地震动分析
                        Sa_current = 0.5 * (Sa_l + Sa_r)  # 用于下一次计算的地震强度
                else:
                    # 不追踪倒塌
                    Sa_current += Sa_incr
                _ida_write_record(self.main.Output_dir, checkpoint_name, signature, gm_name, {
                    'status': 'running',
                    'reason': None,
                    'completed_points': completed_points,
                    'tasks': tasks,
                    'next_run_num': run_num + 1,
                    'next_Sa': float(Sa_current),
                    'iter_state': iter_state,
                    'Sa_l': float(Sa_l),
                    'Sa_r': float(Sa_r),
                })
            else:
                # 超过最大计算次数
                if self.main.trace_collapse:
                    self.mainWin.warning = 1
                    self.signal_add_warning.emit(time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time_gm_end)) + '\n')
                    self.signal_add_warning.emit(f'地震动{gm_name}在{max_ana}次分析后未能找到倒塌点！\n')
                    self.main.logger.warning(f'地震动{gm_name}在{max_ana}次分析后未能找到倒塌点！')
                _ida_write_record(self.main.Output_dir, checkpoint_name, signature, gm_name, {
                    'status': 'completed',
                    'reason': 'max_ana_reached',
                    'completed_points': completed_points,
                    'tasks': tasks,
                    'next_run_num': max_ana,
                    'next_Sa': float(Sa_current),
                    'iter_state': iter_state,
                    'Sa_l': float(Sa_l),
                    'Sa_r': float(Sa_r),
                })
            if self.main.script == 'tcl':
                os.remove(self.main.dir_temp / f'temp_running_{self.main.model_name}_{self.mainWin.current_gm}.tcl')
            self.main.logger.success(f'第{idx+1}条地震动计算完成')
        else:
            self.signal_finished.emit(1)


    def run_IDA_parallel_py(self, processes: int):
        self.main.logger.info(f'正在进行多进程IDA计算')
        self.signal_set_ui.emit(('', '', '', '', ''))
        queue = multiprocessing.Manager().Queue()  # 子进程向主进程通信
        ls_paras: list[tuple] = []
        Sa0, Sa_incr, tol, max_ana = self.mainWin.Sa0, self.mainWin.Sa_incr, self.mainWin.tol, self.mainWin.max_ana
        module = import_module(f'models.{self.main.model_name}')
        run_openseespy = getattr(module, 'run_openseespy')
        maxRunTime = self.main.maxRunTime
        Output_dir = self.main.Output_dir
        MainFolder = self.main.Output_dir
        dir_gm = self.main.dir_gm
        suffix = self.main.suffix
        T0 = self.mainWin.T0
        T_range = self.mainWin.T_range
        T = self.main.T
        intensity_measure = self.mainWin.intensity_measure
        trace_collapse = self.main.trace_collapse
        print_result = self.mainWin.print_result
        collapse_limit = self.main.collapse_limit
        checkpoint_name = getattr(self.main, 'IDA_checkpoint_name', '_ida_checkpoint.json')
        resume = getattr(self.main, 'IDA_resume', True)
        signature = _ida_signature(
            self.main.model_name, self.main.GM_names, T0, Sa0, Sa_incr, tol, max_ana,
            intensity_measure, T_range, trace_collapse, collapse_limit
        )
        _ida_init_summary(Output_dir, checkpoint_name, signature)
        for idx in range(self.main.GM_N):
            gm_name: str = self.main.GM_names[idx]
            dt: float = self.main.GM_dts[idx]
            NPTS: int = self.main.GM_NPTS[idx]
            duration: float = self.main.GM_durations[idx]
            fv_duration: float = self.main.fv_duration
            RSA = self.mainWin.RSA[idx]
            paras = (queue, self.stop_event, run_openseespy, gm_name, dt, NPTS, duration, fv_duration, maxRunTime,
                     Output_dir, MainFolder, dir_gm, suffix, T0, T_range, T, RSA,
                     intensity_measure, Sa0, Sa_incr, tol, max_ana, trace_collapse, print_result,
                     collapse_limit, checkpoint_name, signature, resume)
            ls_paras.append(paras)
        pool = None
        try:
            pool = multiprocessing.Pool(processes, initializer=_ignore_keyboard_interrupt)
            results = []
            for i in range(self.main.GM_N):
                result = pool.apply_async(run_single_IDA_py, ls_paras[i])  # 设置进程池
                results.append(result)
            self.get_queue(queue)
            for result in results:
                output = result.get()
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            self.stop_event.set()
            if pool is not None:
                pool.terminate()
                pool.join()
            _ida_refresh_summary(Output_dir, checkpoint_name, signature)
            self.signal_add_warning.emit('IDA parallel analysis interrupted; unfinished tasks can be resumed.\n')
            self.signal_finished.emit(2)
            return
        except Exception:
            self.stop_event.set()
            if pool is not None:
                pool.terminate()
                pool.join()
            _ida_refresh_summary(Output_dir, checkpoint_name, signature)
            raise
        _ida_refresh_summary(Output_dir, checkpoint_name, signature)
        self.signal_finished.emit(1)
    

    def run_IDA_parallel_tcl(self, processes: int):
        self.main.logger.info(f'正在进行多进程IDA计算')
        self.signal_set_ui.emit(('', '', '', '', ''))
        queue = multiprocessing.Manager().Queue()  # 子进程向主进程通信
        ls_paras: list[tuple] = []
        Sa0, Sa_incr, tol, max_ana = self.mainWin.Sa0, self.mainWin.Sa_incr, self.mainWin.tol, self.mainWin.max_ana
        dir_model = self.main.dir_model
        dir_gm = self.main.dir_gm
        dir_subroutines = self.main.dir_subroutines
        dir_temp = self.main.dir_temp
        model_name = self.main.model_name
        mpco = self.main.mpco
        OS_path = self.main.OS_path
        print_result = self.mainWin.print_result
        maxRunTime = self.main.maxRunTime
        Output_dir = self.main.Output_dir
        dir_gm = self.main.dir_gm
        suffix = self.main.suffix
        T0 = self.mainWin.T0
        T_range = self.mainWin.T_range
        T = self.main.T
        intensity_measure = self.mainWin.intensity_measure
        trace_collapse = self.main.trace_collapse
        collapse_limit = self.main.collapse_limit
        checkpoint_name = getattr(self.main, 'IDA_checkpoint_name', '_ida_checkpoint.json')
        resume = getattr(self.main, 'IDA_resume', True)
        signature = _ida_signature(
            self.main.model_name, self.main.GM_names, T0, Sa0, Sa_incr, tol, max_ana,
            intensity_measure, T_range, trace_collapse, collapse_limit
        )
        _ida_init_summary(Output_dir, checkpoint_name, signature)
        for idx in range(self.main.GM_N):
            gm_name: str = self.main.GM_names[idx]
            dt: float = self.main.GM_dts[idx]
            NPTS: int = self.main.GM_NPTS[idx]
            duration: float = self.main.GM_durations[idx]
            fv_duration: float = self.main.fv_duration
            RSA = self.mainWin.RSA[idx]
            paras = (queue, self.stop_event, dir_model, dir_gm, dir_subroutines, dir_temp,
                     model_name, gm_name, mpco, dt, NPTS, duration, fv_duration, maxRunTime,
                     Output_dir, suffix, T0, T_range, T, RSA, intensity_measure,
                     Sa0, Sa_incr, tol, max_ana, trace_collapse, OS_path, print_result,
                     collapse_limit, checkpoint_name, signature, resume)
            ls_paras.append(paras)
        pool = None
        try:
            pool = multiprocessing.Pool(processes, initializer=_ignore_keyboard_interrupt)
            for i in range(self.main.GM_N):
                pool.apply_async(run_single_IDA_tcl, ls_paras[i])  # 设置进程池
            self.get_queue(queue)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            self.stop_event.set()
            if pool is not None:
                pool.terminate()
                pool.join()
            _ida_refresh_summary(Output_dir, checkpoint_name, signature)
            self.signal_add_warning.emit('IDA parallel analysis interrupted; unfinished tasks can be resumed.\n')
            self.signal_finished.emit(2)
            return
        except Exception:
            self.stop_event.set()
            if pool is not None:
                pool.terminate()
                pool.join()
            _ida_refresh_summary(Output_dir, checkpoint_name, signature)
            raise
        _ida_refresh_summary(Output_dir, checkpoint_name, signature)
        self.signal_finished.emit(1)


    def run_pushover(self):
        self.main.logger.info(f'正在Pushover分析')
        self.signal_set_progressBar.emit(('正在进行Pushover分析...', 0))
        gm_name = 'Pushover'
        dt = '0'
        NPTS = '0'
        duration = '0'
        fv_duration = 0
        SF = 1
        status = 1
        self.signal_set_ui.emit((gm_name, '（仅IDA适用）', duration, dt, NPTS))
        time_gm_start = time.time()
        self.signal_add_log.emit(f'开始：{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time_gm_start))}\n')
        if self.main.script == 'tcl':
            self.modify_script1(self.main.Output_dir, gm_name, dt, NPTS, duration, fv_duration, SF)
            cmd = f'"{self.OS_path}" "{self.main.dir_temp}/temp_running_{self.main.model_name}_{gm_name}.tcl"'
            # 运行分析
            if self.mainWin.print_result:
                subprocess.call(cmd)
            else:
                subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            module = import_module(f'models.{self.main.model_name}')
            run_openseespy = getattr(module, 'run_openseespy')
            maxRunTime = self.main.maxRunTime
            analysis_type = 'PO'
            if self.main.display:
                ShowAnimation = True
            else:
                ShowAnimation = False
            if self.main.mpco:
                MPCO = True
            else:
                MPCO = False
            MainFolder = self.main.Output_dir
            GMname = gm_name
            SubFolder = gm_name
            GMdt = dt
            GMpoints = NPTS
            GMduration = duration
            FVduration = fv_duration
            EqSF = SF
            GMFile = self.main.dir_gm / f'{gm_name}{self.main.suffix}'
            maxRoofDrift = self.main.maxRoofDrift
            RDR_path = [1,2,3,4]    
            paras = [maxRunTime, analysis_type, ShowAnimation, MPCO, MainFolder, GMname, SubFolder,
                        GMdt, GMpoints, GMduration, FVduration, EqSF, GMFile, maxRoofDrift, None,RDR_path]
            self.signal_send_openseespy_paras.emit(paras)
            result = run_openseespy(*paras)
            status = result[0]
            if result[0] == 2:
                self.signal_add_warning.emit(f'{gm_name}分析不收敛')
            elif result[0] == 3:
                self.signal_add_warning.emit(f'{gm_name}超过最大分析时间')
        if not os.path.exists(self.main.Output_dir / gm_name):
            os.makedirs(self.main.Output_dir / gm_name)
        with open(self.main.Output_dir / gm_name / 'isCollapsed.dat', 'w') as f:
            f.write('2')            
        with open(self.main.Output_dir / gm_name / 'Status.dat', 'w') as f:
            f.write(str(status))
        time_gm_end = time.time()
        elapsed_time = time_gm_end - time_gm_start
        self.signal_add_log.emit(f'结束：{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time_gm_end))}\n')
        self.signal_add_log.emit(f'耗时：{round(elapsed_time, 2)}s\n\n')
        self.signal_finished.emit(1)
        if self.main.script == 'tcl':
            os.remove(self.main.dir_temp / f'temp_running_{self.main.model_name}_Pushover.tcl')

    def run_cyclic_pushover(self):
        self.main.logger.info(f'正在Cyclic pushover分析')
        self.signal_set_progressBar.emit(('正在进行Cyclic pushover分析...', 0))
        gm_name = 'Cyclic_pushover'
        dt = '0'
        NPTS = '0'
        duration = '0'
        fv_duration = 0
        SF = 1
        self.signal_set_ui.emit((gm_name, '（仅IDA适用）', duration, dt, NPTS))
        time_gm_start = time.time()
        self.signal_add_log.emit(f'开始：{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time_gm_start))}\n')
        if self.main.script == 'tcl':
            self.modify_script1(self.main.Output_dir, gm_name, dt, NPTS, duration, fv_duration, SF)
            cmd = f'"{self.OS_path}" "{self.main.dir_temp}/temp_running_{self.main.model_name}_{gm_name}.tcl"'
            # 运行分析
            if self.mainWin.print_result:
                subprocess.call(cmd)
            else:
                subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            module = import_module(f'models.{self.main.model_name}')
            run_openseespy = getattr(module, 'run_openseespy')
            maxRunTime = self.main.maxRunTime
            analysis_type = 'CP'
            if self.main.display:
                ShowAnimation = True
            else:
                ShowAnimation = False
            if self.main.mpco:
                MPCO = True
            else:
                MPCO = False
            MainFolder = self.main.Output_dir
            GMname = gm_name
            SubFolder = gm_name
            GMdt = dt
            GMpoints = NPTS
            GMduration = duration
            FVduration = fv_duration
            EqSF = SF
            GMFile = self.main.dir_gm / f'{gm_name}{self.main.suffix}'
            maxRoofDrift = self.main.maxRoofDrift
            RDR_path = self.main.RDR_path
            paras = [maxRunTime, analysis_type, ShowAnimation, MPCO, MainFolder, GMname, SubFolder,
                        GMdt, GMpoints, GMduration, FVduration, EqSF, GMFile, maxRoofDrift, None, RDR_path]
            self.signal_send_openseespy_paras.emit(paras)
            result = run_openseespy(*paras)
            status = result[0] if isinstance(result, (list, tuple)) else result
            if status == 2:
                self.signal_add_warning.emit(f'{gm_name}分析不收敛')
            elif status == 3:
                self.signal_add_warning.emit(f'{gm_name}超过最大分析时间')
        if not os.path.exists(self.main.Output_dir / gm_name):
            os.makedirs(self.main.Output_dir / gm_name)
        with open(self.main.Output_dir / gm_name / 'isCollapsed.dat', 'w') as f:
            f.write('2')            
        time_gm_end = time.time()
        elapsed_time = time_gm_end - time_gm_start
        self.signal_add_log.emit(f'结束：{time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time_gm_end))}\n')
        self.signal_add_log.emit(f'耗时：{round(elapsed_time, 2)}s\n\n')
        self.signal_finished.emit(1)
        if self.main.script == 'tcl':
            os.remove(self.main.dir_temp / f'temp_running_{self.main.model_name}_Cyclic_pushover.tcl')


    @staticmethod
    def Sa(T: np.ndarray, S: np.ndarray, T0: float, withIdx=False) -> float:
        if len(T) == 1 and np.isclose(T[0], T0):
            if withIdx:
                return S[0], 0
            return S[0]
        for i in range(len(T) - 1):
            if T[i] <= T0 <= T[i+1]:
                k = (S[i+1] - S[i]) / (T[i+1] - T[i])
                S0 = S[i] + k * (T0 - T[i])
                if withIdx:
                    return S0, i
                else:
                    return S0
        else:
            raise ValueError(f'无法找到周期点{T0}对应的加速度谱值！')
        
        
    @staticmethod
    def geometric_mean(data):  # 计算几何平均数
        total = 1
        for i in data:
            total *= i
        return pow(total, 1 / len(data))


def Sa(T: np.ndarray, S: np.ndarray, T0: float, withIdx=False) -> float:
    if len(T) == 1 and np.isclose(T[0], T0):
        if withIdx:
            return S[0], 0
        return S[0]
    for i in range(len(T) - 1):
        if T[i] <= T0 <= T[i+1]:
            k = (S[i+1] - S[i]) / (T[i+1] - T[i])
            S0 = S[i] + k * (T0 - T[i])
            if withIdx:
                return S0, i
            else:
                return S0
    else:
        raise ValueError(f'无法找到周期点{T0}对应的加速度谱值！')


def geometric_mean(data):  # 计算几何平均数
    total = 1
    for i in data:
        total *= i
    return pow(total, 1 / len(data))
    

def _ignore_keyboard_interrupt():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _ida_checkpoint_paths(output_dir: Path, checkpoint_name: str):
    checkpoint_name = checkpoint_name or '_ida_checkpoint.json'
    root = Path(output_dir) / f'{Path(checkpoint_name).stem}_records'
    summary = Path(output_dir) / checkpoint_name
    return root, summary


def _atomic_write_json(path: Path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    last_error = None
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass
    raise last_error


def _load_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _ida_signature(model_name: str, gm_names: list[str], T0: float, Sa0: float, Sa_incr: float,
                   tol: float, max_ana: int, intensity_measure: int, T_range, trace_collapse: bool,
                   collapse_limit: float):
    return {
        'model_name': model_name,
        'gm_names': list(gm_names),
        'T0': float(T0),
        'Sa0': float(Sa0),
        'Sa_incr': float(Sa_incr),
        'tol': float(tol),
        'max_ana': int(max_ana),
        'intensity_measure': int(intensity_measure),
        'T_range': list(T_range) if T_range is not None else None,
        'trace_collapse': bool(trace_collapse),
        'collapse_limit': float(collapse_limit),
    }


def _ida_init_summary(output_dir: Path, checkpoint_name: str, signature: dict):
    record_root, summary_path = _ida_checkpoint_paths(output_dir, checkpoint_name)
    record_root.mkdir(parents=True, exist_ok=True)
    summary = _load_json(summary_path, {})
    if summary.get('signature') != signature:
        summary = {
            'format': 'IDA-checkpoint-v1',
            'signature': signature,
            'records': {},
            'updated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        _atomic_write_json(summary_path, summary)
    return record_root, summary_path


def _ida_load_record(output_dir: Path, checkpoint_name: str, signature: dict, gm_name: str):
    record_root, _ = _ida_checkpoint_paths(output_dir, checkpoint_name)
    data = _load_json(record_root / f'{gm_name}.json', None)
    if not data or data.get('signature') != signature:
        return None
    return data


def _ida_write_record(output_dir: Path, checkpoint_name: str, signature: dict, gm_name: str,
                      record: dict, refresh_summary: bool = False):
    record_root, _ = _ida_checkpoint_paths(output_dir, checkpoint_name)
    payload = {
        'format': 'IDA-record-checkpoint-v1',
        'signature': signature,
        'gm_name': gm_name,
        'updated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        **record,
    }
    _atomic_write_json(record_root / f'{gm_name}.json', payload)
    if refresh_summary:
        _ida_refresh_summary(output_dir, checkpoint_name, signature)


def _ida_refresh_summary(output_dir: Path, checkpoint_name: str, signature: dict):
    record_root, summary_path = _ida_checkpoint_paths(output_dir, checkpoint_name)
    records = {}
    if record_root.exists():
        for record_file in record_root.glob('*.json'):
            data = _load_json(record_file, None)
            if data and data.get('signature') == signature:
                tasks = data.get('tasks', {})
                records[data.get('gm_name', record_file.stem)] = {
                    'status': data.get('status'),
                    'next_run_num': data.get('next_run_num'),
                    'next_Sa': data.get('next_Sa'),
                    'last_safe_Sa': data.get('Sa_l'),
                    'first_collapse_Sa': None if data.get('Sa_r') == 100000 else data.get('Sa_r'),
                    'completed_points': len(data.get('completed_points', [])),
                    'task_total': len(tasks),
                    'task_completed': sum(1 for task in tasks.values() if task.get('status') == 'completed'),
                    'task_running': sum(1 for task in tasks.values() if task.get('status') == 'running'),
                    'task_failed': sum(1 for task in tasks.values() if task.get('status') == 'failed'),
                    'task_pending': sum(1 for task in tasks.values() if task.get('status') == 'pending'),
                    'reason': data.get('reason'),
                    'updated_at': data.get('updated_at'),
                }
    summary = {
        'format': 'IDA-checkpoint-v1',
        'signature': signature,
        'records': records,
        'updated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    _atomic_write_json(summary_path, summary)


def _ida_point_complete(output_dir: Path, folder_name: str):
    folder = Path(output_dir) / folder_name
    return (
        folder.exists()
        and (folder / 'Sa.dat').exists()
        and (folder / 'isCollapsed.dat').exists()
        and (folder / 'Sa.dat').stat().st_size > 0
        and (folder / 'isCollapsed.dat').stat().st_size > 0
    )


def _ida_task_key(run_num: int):
    return f'run_{int(run_num)}'


def _ida_read_scalar(path: Path, default=None):
    try:
        data = np.loadtxt(path)
        return float(np.asarray(data).reshape(-1)[0])
    except Exception:
        return default


def _ida_read_collapsed(path: Path, default=None):
    value = _ida_read_scalar(path, default)
    if value is default:
        return default
    return bool(int(round(value)))


def _ida_task_from_output(output_dir: Path, gm_name: str, run_num: int, existing: dict | None = None):
    folder_name = f'{gm_name}_{run_num}'
    task = {
        'task_id': f'{gm_name}_{run_num}',
        'gm_name': gm_name,
        'run_num': int(run_num),
        'folder': folder_name,
        'status': 'pending',
    }
    if existing:
        task.update(existing)
    task.update({
        'task_id': f'{gm_name}_{run_num}',
        'gm_name': gm_name,
        'run_num': int(run_num),
        'folder': folder_name,
    })
    if _ida_point_complete(output_dir, folder_name):
        task['status'] = 'completed'
        task['Sa'] = _ida_read_scalar(Path(output_dir) / folder_name / 'Sa.dat')
        task['collapsed'] = _ida_read_collapsed(Path(output_dir) / folder_name / 'isCollapsed.dat')
        task.setdefault('result_status', 'ok')
    elif task.get('status') in {'running', 'completed'}:
        task['status'] = 'failed'
        task['error'] = 'missing_or_incomplete_output'
    return task


def _ida_build_tasks(output_dir: Path, gm_name: str, completed_points: list[dict],
                     max_ana: int, existing_tasks: dict | None = None):
    existing_tasks = existing_tasks or {}
    tasks = {}
    last_seen = min(max_ana, max(len(completed_points), len(existing_tasks)))
    for run_num in range(1, last_seen + 1):
        key = _ida_task_key(run_num)
        existing = existing_tasks.get(key)
        if not existing:
            for point in completed_points:
                if int(point.get('run_num', 0)) == run_num:
                    existing = {
                        'status': 'completed',
                        'Sa': point.get('Sa'),
                        'SF': point.get('SF'),
                        'collapsed': point.get('collapsed'),
                        'elapsed_seconds': point.get('elapsed_seconds'),
                        'result_status': point.get('result_status', 'ok'),
                    }
                    break
        tasks[key] = _ida_task_from_output(output_dir, gm_name, run_num, existing)
    return tasks


def _ida_rebuild_state_from_outputs(output_dir: Path, gm_name: str, Sa0: float, Sa_incr: float,
                                    tol: float, max_ana: int, trace_collapse: bool,
                                    existing_tasks: dict | None = None):
    completed_points = []
    tasks = {}
    Sa_current = float(Sa0)
    iter_state = 0
    Sa_l = 0.0
    Sa_r = 100000.0
    status = 'running'
    reason = None
    for run_num in range(1, max_ana + 1):
        task = _ida_task_from_output(output_dir, gm_name, run_num, (existing_tasks or {}).get(_ida_task_key(run_num)))
        if task.get('status') != 'completed':
            break
        Sa_done = float(task.get('Sa', round(Sa_current, 5)))
        collapsed = bool(task.get('collapsed'))
        point = {
            'run_num': run_num,
            'Sa': Sa_done,
            'SF': task.get('SF'),
            'collapsed': collapsed,
            'elapsed_seconds': task.get('elapsed_seconds'),
            'folder': task.get('folder', f'{gm_name}_{run_num}'),
            'result_status': task.get('result_status', 'ok'),
        }
        completed_points.append(point)
        tasks[_ida_task_key(run_num)] = task
        if trace_collapse:
            if run_num == 1 and collapsed:
                status = 'completed'
                reason = 'first_run_collapse'
                Sa_r = Sa_done
                Sa_current = Sa_done
                break
            if not collapsed and iter_state == 0:
                Sa_l = Sa_done
                Sa_current = Sa_done + Sa_incr
            else:
                iter_state = 1
                if collapsed:
                    Sa_r = min(Sa_done, Sa_r)
                else:
                    Sa_l = max(Sa_done, Sa_l)
                if Sa_r - Sa_l < tol:
                    status = 'completed'
                    reason = 'collapse_tolerance_reached'
                    Sa_current = Sa_done
                    break
                Sa_current = 0.5 * (Sa_l + Sa_r)
        else:
            Sa_current = Sa_done + Sa_incr
    next_run_num = len(completed_points)
    if status != 'completed' and next_run_num >= max_ana:
        status = 'completed'
        reason = 'max_ana_reached'
    return {
        'status': status,
        'reason': reason,
        'completed_points': completed_points,
        'tasks': tasks,
        'next_run_num': next_run_num,
        'next_Sa': float(Sa_current),
        'iter_state': iter_state,
        'Sa_l': float(Sa_l),
        'Sa_r': float(Sa_r),
    }


def _ida_resume_state(output_dir: Path, checkpoint_name: str, signature: dict, gm_name: str,
                      Sa0: float, Sa_incr: float, tol: float, max_ana: int,
                      trace_collapse: bool, resume: bool):
    default = {
        'status': 'running',
        'completed_points': [],
        'tasks': {},
        'next_run_num': 0,
        'next_Sa': float(Sa0),
        'iter_state': 0,
        'Sa_l': 0.0,
        'Sa_r': 100000.0,
    }
    if not resume:
        return default
    checkpoint = _ida_load_record(output_dir, checkpoint_name, signature, gm_name)
    if not checkpoint:
        rebuilt = _ida_rebuild_state_from_outputs(output_dir, gm_name, Sa0, Sa_incr, tol, max_ana, trace_collapse)
        if rebuilt['completed_points'] or rebuilt['tasks']:
            _ida_write_record(output_dir, checkpoint_name, signature, gm_name, rebuilt)
            return {**default, **rebuilt}
        return default
    completed = checkpoint.get('completed_points', [])
    valid_points = []
    for idx, point in enumerate(completed):
        if idx >= max_ana:
            break
        if not _ida_point_complete(output_dir, f'{gm_name}_{idx + 1}'):
            break
        valid_points.append(point)
    if len(valid_points) != len(completed) or not checkpoint.get('tasks'):
        rebuilt = _ida_rebuild_state_from_outputs(
            output_dir, gm_name, Sa0, Sa_incr, tol, max_ana, trace_collapse, checkpoint.get('tasks')
        )
        checkpoint.update(rebuilt)
        _ida_write_record(output_dir, checkpoint_name, signature, gm_name, checkpoint)
        return {**default, **checkpoint}
    checkpoint['completed_points'] = valid_points
    checkpoint['tasks'] = _ida_build_tasks(output_dir, gm_name, valid_points, max_ana, checkpoint.get('tasks'))
    checkpoint['next_run_num'] = len(valid_points)
    if len(valid_points) != len(completed):
        checkpoint['status'] = 'running'
        _ida_write_record(output_dir, checkpoint_name, signature, gm_name, checkpoint)
    if int(checkpoint.get('next_run_num', 0)) >= max_ana:
        checkpoint['status'] = 'completed'
        checkpoint['reason'] = checkpoint.get('reason') or 'max_ana_reached'
        _ida_write_record(output_dir, checkpoint_name, signature, gm_name, checkpoint)
    if checkpoint.get('status') == 'completed':
        expected_points = int(checkpoint.get('next_run_num', len(valid_points)))
        completed_tasks = [
            task for task in checkpoint.get('tasks', {}).values()
            if task.get('status') == 'completed'
        ]
        if len(completed_tasks) < expected_points:
            rebuilt = _ida_rebuild_state_from_outputs(
                output_dir, gm_name, Sa0, Sa_incr, tol, max_ana, trace_collapse, checkpoint.get('tasks')
            )
            checkpoint.update(rebuilt)
            _ida_write_record(output_dir, checkpoint_name, signature, gm_name, checkpoint)
    return {**default, **checkpoint}


def run_single_IDA_py(
    queue: multiprocessing.Queue,
    stop_event,
    run_openseespy: Callable,
    gm_name: str,
    dt: float,
    NPTS: int,
    duration: float,
    fv_duration: float,
    maxRunTime: float,
    Output_dir: Path,
    MainFolder: Path,
    dir_gm: Path,
    suffix: str,
    T0: float,
    T_range: tuple[float, float],
    T: np.ndarray,
    RSA: np.ndarray,
    intensity_measure: Literal[1, 2],
    Sa0: float,
    Sa_incr: float,
    tol: float,
    max_ana: int,
    trace_collapse: bool,
    print_result: bool,
    collapse_limit: float,
    checkpoint_name: str = '_ida_checkpoint.json',
    checkpoint_signature: dict | None = None,
    resume: bool = True,
):
    """
    计算单条地震动的IDA分析(基于openseespy)，
    该函数仅用于多进程，无法详细地更新ui监控信息。
    队列格式：(信息类型，信息值)
    信息类型：
    * a-该条地震动IDA开始  
    * b-该条地震动IDA结束
    * c-第xx次计算开始
    * d-第xx次计算结束
    * e-首次计算即倒塌
    * f-超过最大计算次数仍未找到倒塌点
    * g-不收敛
    * h-超过最大计算时间
    """
    try:
        now = lambda: datetime.datetime.now().strftime('%H:%M')
        message = ('a', f'{gm_name}开始({now()})\n')
        if not stop_event.is_set():
            queue.put(message)
        Sa_current = Sa0
        if intensity_measure == 1:
            Sa_original = Sa(T, RSA, T0)  # 以Sa(T)作为地震动强度指标
        elif intensity_measure == 2:
            Ta, Tb = T_range
            Sa_range = RSA[(Ta <= T) & (T <= Tb)]
            Sa_avg = geometric_mean(Sa_range)  # 简单几何平均数
            Sa_original = Sa_avg  
        if checkpoint_signature is None:
            checkpoint_signature = _ida_signature(
                '', [gm_name], T0, Sa0, Sa_incr, tol, max_ana,
                intensity_measure, T_range, trace_collapse, collapse_limit
            )
        state = _ida_resume_state(
            Output_dir, checkpoint_name, checkpoint_signature, gm_name,
            Sa0, Sa_incr, tol, max_ana, trace_collapse, resume
        )
        if state.get('status') == 'completed':
            queue.put(('b', f'{gm_name} checkpoint completed, skip({now()})\n'))
            return
        Sa_current = float(state.get('next_Sa', Sa0))
        iter_state = int(state.get('iter_state', 0))
        Sa_l = float(state.get('Sa_l', 0))
        Sa_r = float(state.get('Sa_r', 100000))
        completed_points = list(state.get('completed_points', []))
        tasks = dict(state.get('tasks', {}))
        start_run_num = int(state.get('next_run_num', len(completed_points)))
        if start_run_num > 0:
            queue.put(('a', f'{gm_name} resume from checkpoint: completed {start_run_num}, next Sa={Sa_current}g\n'))
        for run_num in range(start_run_num, max_ana):
            if stop_event.is_set():
                message = ('b', f'{gm_name}退出计算\n')
                queue.put(message)
                return
            Sa_current = round(Sa_current, 5)
            SF = Sa_current / Sa_original
            task_key = _ida_task_key(run_num + 1)
            tasks[task_key] = {
                'task_id': f'{gm_name}_{run_num + 1}',
                'gm_name': gm_name,
                'run_num': run_num + 1,
                'folder': f'{gm_name}_{run_num + 1}',
                'status': 'running',
                'Sa': float(Sa_current),
                'SF': float(SF),
                'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
            }
            _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                'status': 'running',
                'reason': None,
                'completed_points': completed_points,
                'tasks': tasks,
                'next_run_num': run_num,
                'next_Sa': float(Sa_current),
                'iter_state': iter_state,
                'Sa_l': float(Sa_l),
                'Sa_r': float(Sa_r),
            })
            message = ('c', f'{gm_name}第{run_num+1}次计算开始_Sa={Sa_current}\n')
            queue.put(message)
            time_gm_start = time.time()
            # maxRunTime
            analysis_type = 'TH'
            ShowAnimation = False
            MPCO = False
            # MainFolder
            GMname = gm_name
            SubFolder = f'{gm_name}_{run_num+1}'
            GMdt = dt
            GMpoints = NPTS
            GMduration = duration
            FVduration = fv_duration
            EqSF = SF
            GMFile = dir_gm / f'{gm_name}{suffix}'
            maxRoofDrift = 0.1
            paras = [maxRunTime, analysis_type, ShowAnimation, MPCO, MainFolder, GMname, SubFolder,
                    GMdt, GMpoints, GMduration, FVduration, EqSF, GMFile, maxRoofDrift, collapse_limit, None]
            with HiddenPrints(not print_result):
                result = run_openseespy(*paras)  # 运行分析
            result_status = 'ok'
            if result[2]:
                collapsed = 1  # 分析完成，倒塌
            else:
                collapsed = 0  # 分析完成，未倒塌
            if result[0] == 2:
                s = '倒塌'if collapsed else '未倒塌'
                message = ('g', f'{gm_name}第{run_num+1}次计算不收敛({s})\n')
                queue.put(message)
            elif result[0] == 3:
                message = ('h', f'{gm_name}第{run_num+1}次计算超过最大计算时间\n')
                queue.put(message)
            if not os.path.exists(Output_dir / f'{gm_name}_{run_num+1}'):
                os.makedirs(Output_dir / f'{gm_name}_{run_num+1}')
            np.savetxt(Output_dir / f'{gm_name}_{run_num+1}/Sa.dat', np.array([Sa_current]))
            with open(Output_dir / f'{gm_name}_{run_num+1}/isCollapsed.dat', 'w') as f:
                f.write(str(collapsed))
            time_gm_end = time.time()
            time_cost = time_gm_end - time_gm_start
            point = {
                'run_num': run_num + 1,
                'Sa': float(Sa_current),
                'SF': float(SF),
                'collapsed': bool(collapsed),
                'elapsed_seconds': round(time_cost, 3),
                'folder': f'{gm_name}_{run_num + 1}',
                'result_status': result_status,
            }
            completed_points.append(point)
            tasks[task_key] = {
                **tasks.get(task_key, {}),
                'status': 'completed',
                'collapsed': bool(collapsed),
                'elapsed_seconds': round(time_cost, 3),
                'result_status': result_status,
                'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
            }
            if trace_collapse:
                # 追踪倒塌
                if run_num == 0 and collapsed == 1:
                    # self.signal_add_warning.emit(f'{gm_name}首次计算即倒塌！\n\n')
                    message = ('e', f'{gm_name}首次计算即倒塌\n')
                    queue.put(message)
                    _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                        'status': 'completed',
                        'reason': 'first_run_collapse',
                        'completed_points': completed_points,
                        'tasks': tasks,
                        'next_run_num': run_num + 1,
                        'next_Sa': float(Sa_current),
                        'iter_state': iter_state,
                        'Sa_l': float(Sa_l),
                        'Sa_r': float(Sa_current),
                    })
                    return
                if collapsed == 0 and iter_state == 0:
                    # 如果未倒塌，且不处于迭代状态
                    Sa_l = Sa_current
                    Sa_current += Sa_incr
                else:
                    # 在迭代状态下，即已经出现过倒塌后
                    iter_state = 1  # 进入迭代状态
                    if collapsed == 1:
                        # 若迭代状态下倒塌，更新最小倒塌强度
                        Sa_r = min(Sa_current, Sa_r)
                    else:
                        # 若迭代状态下未倒塌，更新最大未倒塌强度
                        Sa_l = max(Sa_current, Sa_l)
                    if Sa_l > Sa_r:
                        raise ValueError('最小倒塌强度大于最大未倒塌强度!')
                    if Sa_r - Sa_l < tol:
                        s = '倒塌' if collapsed else '未倒塌'
                        _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                            'status': 'completed',
                            'reason': 'collapse_tolerance_reached',
                            'completed_points': completed_points,
                            'tasks': tasks,
                            'next_run_num': run_num + 1,
                            'next_Sa': float(Sa_current),
                            'iter_state': iter_state,
                            'Sa_l': float(Sa_l),
                            'Sa_r': float(Sa_r),
                        })
                        message = ('d', f'{gm_name}第{run_num+1}次计算完成_{s}\n')
                        queue.put(message)
                        message = ('b', f'{gm_name}完成({now()})\n')
                        queue.put(message)
                        return
                    Sa_current = 0.5 * (Sa_l + Sa_r)  # 用于下一次计算的地震强度
            else:
                # 不追踪倒塌
                Sa_current += Sa_incr
            _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                'status': 'running',
                'reason': None,
                'completed_points': completed_points,
                'tasks': tasks,
                'next_run_num': run_num + 1,
                'next_Sa': float(Sa_current),
                'iter_state': iter_state,
                'Sa_l': float(Sa_l),
                'Sa_r': float(Sa_r),
            })
            s = '倒塌'if collapsed else '未倒塌'
            message = ('d', f'{gm_name}第{run_num+1}次计算完成_{s}\n')
            queue.put(message)
        else:
            # 超过最大计算次数
            if trace_collapse:
                message = ('f', f'{gm_name}超过最大计算次数仍未找到倒塌点\n')
                queue.put(message)
            _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                'status': 'completed',
                'reason': 'max_ana_reached',
                'completed_points': completed_points,
                'tasks': tasks,
                'next_run_num': max_ana,
                'next_Sa': float(Sa_current),
                'iter_state': iter_state,
                'Sa_l': float(Sa_l),
                'Sa_r': float(Sa_r),
            })
            message = ('b', f'{gm_name}完成({now()})\n')
            queue.put(message)
            return
    except Exception as e:
        if 'task_key' in locals() and 'tasks' in locals():
            tasks[task_key] = {
                **tasks.get(task_key, {}),
                'status': 'failed',
                'error': repr(e),
                'failed_at': datetime.datetime.now().isoformat(timespec='seconds'),
            }
            try:
                _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                    'status': 'running',
                    'reason': None,
                    'completed_points': completed_points if 'completed_points' in locals() else [],
                    'tasks': tasks,
                    'next_run_num': run_num if 'run_num' in locals() else 0,
                    'next_Sa': float(Sa_current) if 'Sa_current' in locals() else float(Sa0),
                    'iter_state': iter_state if 'iter_state' in locals() else 0,
                    'Sa_l': float(Sa_l) if 'Sa_l' in locals() else 0.0,
                    'Sa_r': float(Sa_r) if 'Sa_r' in locals() else 100000.0,
                })
            except Exception:
                pass
        tb = traceback.format_exc()
        queue.put(('i', (e, tb)))

def run_single_IDA_tcl(
    queue: multiprocessing.Queue,
    stop_event,
    dir_model: Path,
    dir_gm: Path,
    dir_subroutines: Path,
    dir_temp: Path,
    model_name: str,
    gm_name: str,
    mpco: bool,
    dt: float,
    NPTS: int,
    duration: float,
    fv_duration: float,
    maxRunTime: float,
    Output_dir: Path,
    suffix: str,
    T0: float,
    T_range: tuple[float, float],
    T: np.ndarray,
    RSA: np.ndarray,
    intensity_measure: Literal[1, 2],
    Sa0: float,
    Sa_incr: float,
    tol: float,
    max_ana: int,
    trace_collapse: bool,
    OS_path: str,
    print_result: bool,
    collapse_limit: float,
    checkpoint_name: str = '_ida_checkpoint.json',
    checkpoint_signature: dict | None = None,
    resume: bool = True,
):
    """
    计算单条地震动的IDA分析(基于tcl)，
    该函数仅用于多进程，无法详细地更新ui监控信息。
    队列格式：(信息类型，信息值)
    信息类型：
    * a-该条地震动IDA开始  
    * b-该条地震动IDA结束
    * c-第xx次计算开始
    * d-第xx次计算结束
    * e-首次计算即倒塌
    * f-超过最大计算次数仍未找到倒塌点
    * g-不收敛
    * h-超过最大计算时间
    """
    try:
        now = lambda: datetime.datetime.now().strftime('%H:%M')
        message = ('a', f'{gm_name}开始({now()})\n')
        if not stop_event.is_set():
            queue.put(message)
        Sa_current = Sa0
        if intensity_measure == 1:
            Sa_original = Sa(T, RSA, T0)  # 以Sa(T)作为地震动强度指标
        elif intensity_measure == 2:
            Ta, Tb = T_range
            Sa_range = RSA[(Ta <= T) & (T <= Tb)]
            Sa_avg = geometric_mean(Sa_range)  # 简单几何平均数
            Sa_original = Sa_avg  
        if checkpoint_signature is None:
            checkpoint_signature = _ida_signature(
                model_name, [gm_name], T0, Sa0, Sa_incr, tol, max_ana,
                intensity_measure, T_range, trace_collapse, collapse_limit
            )
        state = _ida_resume_state(
            Output_dir, checkpoint_name, checkpoint_signature, gm_name,
            Sa0, Sa_incr, tol, max_ana, trace_collapse, resume
        )
        if state.get('status') == 'completed':
            queue.put(('b', f'{gm_name} checkpoint completed, skip({now()})\n'))
            return
        Sa_current = float(state.get('next_Sa', Sa0))
        iter_state = int(state.get('iter_state', 0))
        Sa_l = float(state.get('Sa_l', 0))
        Sa_r = float(state.get('Sa_r', 100000))
        completed_points = list(state.get('completed_points', []))
        tasks = dict(state.get('tasks', {}))
        start_run_num = int(state.get('next_run_num', len(completed_points)))
        if start_run_num > 0:
            queue.put(('a', f'{gm_name} resume from checkpoint: completed {start_run_num}, next Sa={Sa_current}g\n'))
        for run_num in range(start_run_num, max_ana):
            if stop_event.is_set():
                message = ('b', f'{gm_name}退出计算\n')
                queue.put(message)
                return
            Sa_current = round(Sa_current, 5)
            SF = Sa_current / Sa_original
            message = ('c', f'{gm_name}第{run_num+1}次计算开始_Sa={Sa_current}\n')
            queue.put(message)
            time_gm_start = time.time()
            maxRoofDrift = 0.1
            display = False
            running_case = 'IDA'
            task_key = _ida_task_key(run_num + 1)
            tasks[task_key] = {
                'task_id': f'{gm_name}_{run_num + 1}',
                'gm_name': gm_name,
                'run_num': run_num + 1,
                'folder': f'{gm_name}_{run_num + 1}',
                'status': 'running',
                'Sa': float(Sa_current),
                'SF': float(SF),
                'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
            }
            _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                'status': 'running',
                'reason': None,
                'completed_points': completed_points,
                'tasks': tasks,
                'next_run_num': run_num,
                'next_Sa': float(Sa_current),
                'iter_state': iter_state,
                'Sa_l': float(Sa_l),
                'Sa_r': float(Sa_r),
            })
            WorkerThread.modify_script(
                dir_model, model_name, maxRunTime, running_case,
                dir_gm, dir_subroutines, dir_temp, suffix, display, mpco, maxRoofDrift,
                Output_dir, gm_name, dt, NPTS, duration, fv_duration, SF, collapse_limit, run_num + 1
            )
            cmd = f'"{OS_path}" "{dir_temp}/temp_running_{model_name}_{gm_name}.tcl"'
            if print_result:
                subprocess.call(cmd)
            else:
                subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(dir_temp/ f'{gm_name}_CollapseState.txt'):
                collapsed = 1
                os.remove(dir_temp / f'{gm_name}_CollapseState.txt')
            else:
                collapsed = 0
            if not os.path.exists(Output_dir / f'{gm_name}_{run_num+1}'):
                os.makedirs(Output_dir / f'{gm_name}_{run_num+1}')
                print(Output_dir / f'{gm_name}_{run_num+1}')
            np.savetxt(Output_dir / f'{gm_name}_{run_num+1}/Sa.dat', np.array([Sa_current]))
            with open(Output_dir / f'{gm_name}_{run_num+1}/isCollapsed.dat', 'w') as f:
                f.write(str(collapsed))
            time_gm_end = time.time()
            time_cost = time_gm_end - time_gm_start
            point = {
                'run_num': run_num + 1,
                'Sa': float(Sa_current),
                'SF': float(SF),
                'collapsed': bool(collapsed),
                'elapsed_seconds': round(time_cost, 3),
                'folder': f'{gm_name}_{run_num + 1}',
                'result_status': 'ok',
            }
            completed_points.append(point)
            tasks[task_key] = {
                **tasks.get(task_key, {}),
                'status': 'completed',
                'collapsed': bool(collapsed),
                'elapsed_seconds': round(time_cost, 3),
                'result_status': 'ok',
                'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
            }
            if trace_collapse:
                # 追踪倒塌
                if run_num == 0 and collapsed == 1:
                    # self.signal_add_warning.emit(f'{gm_name}首次计算即倒塌！\n\n')
                    message = ('e', f'{gm_name}首次计算即倒塌\n')
                    queue.put(message)
                    _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                        'status': 'completed',
                        'reason': 'first_run_collapse',
                        'completed_points': completed_points,
                        'tasks': tasks,
                        'next_run_num': run_num + 1,
                        'next_Sa': float(Sa_current),
                        'iter_state': iter_state,
                        'Sa_l': float(Sa_l),
                        'Sa_r': float(Sa_current),
                    })
                    return
                if collapsed == 0 and iter_state == 0:
                    # 如果未倒塌，且不处于迭代状态
                    Sa_l = Sa_current
                    Sa_current += Sa_incr
                else:
                    # 在迭代状态下，即已经出现过倒塌后
                    iter_state = 1  # 进入迭代状态
                    if collapsed == 1:
                        # 若迭代状态下倒塌，更新最小倒塌强度
                        Sa_r = min(Sa_current, Sa_r)
                    else:
                        # 若迭代状态下未倒塌，更新最大未倒塌强度
                        Sa_l = max(Sa_current, Sa_l)
                    if Sa_l > Sa_r:
                        raise ValueError('最小倒塌强度大于最大未倒塌强度!')
                    if Sa_r - Sa_l < tol:
                        s = '倒塌' if collapsed else '未倒塌'
                        _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                            'status': 'completed',
                            'reason': 'collapse_tolerance_reached',
                            'completed_points': completed_points,
                            'tasks': tasks,
                            'next_run_num': run_num + 1,
                            'next_Sa': float(Sa_current),
                            'iter_state': iter_state,
                            'Sa_l': float(Sa_l),
                            'Sa_r': float(Sa_r),
                        })
                        message = ('d', f'{gm_name}第{run_num+1}次计算完成_{s}\n')
                        queue.put(message)
                        message = ('b', f'{gm_name}完成({now()})\n')
                        queue.put(message)
                        return
                    Sa_current = 0.5 * (Sa_l + Sa_r)  # 用于下一次计算的地震强度
            else:
                # 不追踪倒塌
                Sa_current += Sa_incr
            _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                'status': 'running',
                'reason': None,
                'completed_points': completed_points,
                'tasks': tasks,
                'next_run_num': run_num + 1,
                'next_Sa': float(Sa_current),
                'iter_state': iter_state,
                'Sa_l': float(Sa_l),
                'Sa_r': float(Sa_r),
            })
            s = '倒塌'if collapsed else '未倒塌'
            message = ('d', f'{gm_name}第{run_num+1}次计算完成_{s}\n')
            queue.put(message)
        else:
            # 超过最大计算次数
            if trace_collapse:
                message = ('f', f'{gm_name}超过最大计算次数仍未找到倒塌点\n')
                queue.put(message)
                pass
            _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                'status': 'completed',
                'reason': 'max_ana_reached',
                'completed_points': completed_points,
                'tasks': tasks,
                'next_run_num': max_ana,
                'next_Sa': float(Sa_current),
                'iter_state': iter_state,
                'Sa_l': float(Sa_l),
                'Sa_r': float(Sa_r),
            })
            message = ('b', f'{gm_name}完成({now()})\n')
            queue.put(message)
            return
    except Exception as e:
        if 'task_key' in locals() and 'tasks' in locals():
            tasks[task_key] = {
                **tasks.get(task_key, {}),
                'status': 'failed',
                'error': repr(e),
                'failed_at': datetime.datetime.now().isoformat(timespec='seconds'),
            }
            try:
                _ida_write_record(Output_dir, checkpoint_name, checkpoint_signature, gm_name, {
                    'status': 'running',
                    'reason': None,
                    'completed_points': completed_points if 'completed_points' in locals() else [],
                    'tasks': tasks,
                    'next_run_num': run_num if 'run_num' in locals() else 0,
                    'next_Sa': float(Sa_current) if 'Sa_current' in locals() else float(Sa0),
                    'iter_state': iter_state if 'iter_state' in locals() else 0,
                    'Sa_l': float(Sa_l) if 'Sa_l' in locals() else 0.0,
                    'Sa_r': float(Sa_r) if 'Sa_r' in locals() else 100000.0,
                })
            except Exception:
                pass
        tb = traceback.format_exc()
        queue.put(('i', (e, tb)))

def run_single_th(
    queue: multiprocessing.Queue,
    script: Literal['tcl', 'py'],
    dir_model: Path,
    dir_gm: Path,
    dir_subroutines: Path,
    dir_temp: Path,
    model_name: str,
    gm_name: str,
    mpco: bool,
    dt: float,
    NPTS: int,
    duration: float,
    fv_duration: float,
    maxRunTime: float,
    Output_dir: Path,
    suffix: str,
    SF: float,
    method,
    th_para,
    OS_path: str,
    print_result: bool,
    collapse_limit: float,
):
    """
    计算单条地震动的IDA分析(基于tcl)，
    该函数仅用于多进程，无法详细地更新ui监控信息。
    队列格式：(信息类型，信息值)
    信息类型：
    * a-该条地震动IDA开始  
    * b-该条地震动IDA结束
    * c-第xx次计算开始
    * d-第xx次计算结束
    * e-首次计算即倒塌
    * f-超过最大计算次数仍未找到倒塌点
    * g-不收敛
    * h-超过最大计算时间
    """
    try:
        now = lambda: datetime.datetime.now().strftime('%H:%M')
        message = ('a', f'{gm_name}开始({now()})\n')
        queue.put(message)
        time_gm_start = time.time()
        maxRoofDrift = 0.1
        display = False
        running_case = 'TH'
        if script == 'tcl':
            WorkerThread.modify_script(
                dir_model, model_name, maxRunTime, running_case,
                dir_gm, dir_subroutines, dir_temp, suffix, display, mpco, maxRoofDrift,
                Output_dir, gm_name, dt, NPTS, duration, fv_duration, SF, collapse_limit)
            cmd = f'"{OS_path}" "{dir_temp}/temp_running_{model_name}_{gm_name}.tcl"'
            if print_result:
                subprocess.call(cmd)
            else:
                subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(dir_temp/ f'{gm_name}_CollapseState.txt'):
                collapsed = 1
                os.remove(dir_temp / f'{gm_name}_CollapseState.txt')
            else:
                collapsed = 0
        else:
            module = import_module(f'models.{model_name}')
            run_openseespy = getattr(module, 'run_openseespy')
            # maxRunTime
            analysis_type = 'TH'
            ShowAnimation = False
            MPCO = False
            MainFolder = Output_dir
            GMname = gm_name
            SubFolder = f'{gm_name}'
            GMdt = dt
            GMpoints = NPTS
            GMduration = duration
            FVduration = fv_duration
            EqSF = SF
            GMFile = dir_gm / f'{gm_name}{suffix}'
            maxRoofDrift = 0.1
            paras = [maxRunTime, analysis_type, ShowAnimation, MPCO, MainFolder, GMname, SubFolder,
                    GMdt, GMpoints, GMduration, FVduration, EqSF, GMFile, maxRoofDrift, collapse_limit, None]
            with HiddenPrints(not print_result):
                result = run_openseespy(*paras)  # 运行分析
            if result[2]:
                collapsed = 1  # 分析完成，倒塌
            else:
                collapsed = 0  # 分析完成，未倒塌
            if result[0] == 2:
                s = '倒塌'if collapsed else '未倒塌'
                message = ('g', f'{gm_name}计算不收敛({s})\n')
                queue.put(message)
            elif result[0] == 3:
                message = ('h', f'{gm_name}计算超过最大计算时间\n')
                queue.put(message)
        if not os.path.exists(Output_dir / f'{gm_name}'):
            os.makedirs(Output_dir / f'{gm_name}')
        Sa = None
        if method == 'd':
            Sa = th_para  # 指定PGA
        elif method == 'i':
            Sa = th_para[1]  # 指定Sa(Ta)
        elif method == 'j':
            Sa = th_para[2]  # 指定Sa,avg
        if Sa:
            np.savetxt(Output_dir / gm_name / 'Sa.dat', np.array([Sa]))
        with open(Output_dir/ gm_name / 'isCollapsed.dat', 'w') as f:
            f.write(str(collapsed))
        if script == 'tcl':
            os.remove(dir_temp / f'temp_running_{model_name}_{gm_name}.tcl')    
        time_gm_end = time.time()
        time_cost = time_gm_end - time_gm_start
        s = '倒塌'if collapsed else '未倒塌'
        message = ('b', f'{gm_name}完成_{s}\n')
        queue.put(message)
    except Exception as e:
        tb = traceback.format_exc()
        queue.put(('i', (e, tb)))


class HiddenPrints:
    """屏蔽输出"""

    def __init__(self, suppress=True):
        self.suppress = suppress
        self._original_stdout = None
        self._original_stderr = None

    def __enter__(self):
        if self.suppress:
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
