# Paint 绘图脚本说明

> 推荐非编程用户直接使用点击式界面。双击项目根目录的
> `启动绘图助手.bat`，详细说明见 `Paint/README_绘图助手.md`。

本目录按绘图任务保留入口脚本。常用脚本都在文件顶部设置了“用户配置区”，优先直接修改配置区后运行，不再使用交互式输入。

## 使用方式

1. 打开目标脚本。
2. 修改顶部 `CONFIG` 或“用户配置区”中的模型、温度、工况、指标、保存路径等参数。
3. 在项目根目录运行脚本，例如：

```bash
python Paint/scripts/plot_response_profiles.py
```

共享样式由 `plot_common.py` 管理，包括字体、温度符号、项目路径和通用文件名工具。

## 统一入口

- `plot_response_profiles.py`：响应楼层剖面图。合并原层加速度分布、层间位移角剖面、残余层间位移角剖面脚本。支持 `IDR`、`RIDR`、`PFA`，支持 `TH`/`IDA`，支持不同模型对比或同一模型不同温度对比。
- `plot_response_boxplots.py`：响应箱体图。合并原“不同模型/工况响应箱体图”和“温度分组响应箱体图”。支持 `IDR`、`RIDR`、`PFA`、`DCF`，支持按工况分组或按温度分组。
- `plot_story_response.py`：楼层剪力/楼层刚度剖面图。合并原楼层剪力和楼层刚度脚本。支持 `shear`、`stiffness`，支持 `TH` 统计结果和 `PO` Pushover 输出计算，支持不同模型对比或同一模型不同温度对比。
- `plot_hinge_profiles.py`：梁铰/柱铰统计图。合并原梁铰和柱铰脚本。支持 `beam`、`column`，支持按楼层最大塑性转角对比，也支持单个模型的构件位置明细图。

## 保留脚本

- `plot_pushover_curves.py`：Pushover 曲线，支持单模型多温度或多模型对比。
- `plot_ida_curves.py`：IDA 曲线和多 EDP 组合图。
- `plot_fragility.py`：易损性曲线、易损性曲面、温度切片图。
- `plot_regional_fragility_yushu_turpan_wenchang.py`：玉树-吐鲁番-文昌区域易损性。
- `plot_regional_fragility_mohe_turpan_sanya.py`：漠河-吐鲁番-三亚区域易损性。
- `plot_regional_hazard.py`：区域危险性曲线和局部放大图。
- `plot_psdm.py`：PSDM 散点和拟合曲线。
- `summarize_psdm_parameters.py`：PSDM 参数汇总表。
- `plot_sma_support.py`：SMA 支撑滞回曲线和 GIF。
- `plot_time_history.py`：单条记录时程对比。
- `plot_hazard_curve.py`：危险性曲线。
- `plot_collapse_probability.py`：倒塌/超越概率曲线。
- `check_ida_convergence.py`：IDA 收敛统计。
- `export_english_figures.py`：英文图件批量导出。

## 已合并删除的旧入口

- `plot_floor_acceleration.py`：已并入 `plot_response_profiles.py`。
- `plot_interstory_drift_history.py`：已并入 `plot_response_profiles.py`。
- `plot_residual_drift.py`：已并入 `plot_response_profiles.py`。
- `plot_response_box_by_case.py`：已并入 `plot_response_boxplots.py`。
- `plot_story_shear.py`：已并入 `plot_story_response.py`。
- `plot_story_stiffness.py`：已并入 `plot_story_response.py`。
- `plot_hinge_beam.py`：已并入 `plot_hinge_profiles.py`。
- `plot_hinge_column.py`：已并入 `plot_hinge_profiles.py`。

## 路径和文件名约定

新入口脚本使用标准中文文件名：

```text
层间位移角.csv
残余层间位移角.csv
层加速度(g).csv
DCF.csv
楼层剪力_统计.csv
层间位移角_统计.csv
梁铰_统计_50th.csv
柱铰_统计_50th.csv
层间位移角.out
残余层间位移角.out
层加速度(g).out
周期(s).out
```

易损性曲线 Excel 的标准文件名为：

```text
易损性曲线_IDR.xlsx
易损性曲线_RIDR.xlsx
易损性曲线_PFA.xlsx
```

如果旧结果数据中存在乱码文件名，建议先重命名为以上标准名称后再运行新脚本。
