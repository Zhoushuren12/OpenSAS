import os
from os import path
import sys
from pathlib import Path
sys.path.append(Path.cwd().as_posix())
import time
import builtins
original_open = builtins.open
def utf8_open(*args, **kwargs):
    if 'encoding' not in kwargs:
        mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')
        if 'b' not in mode:  
            kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)
builtins.open = utf8_open


def _period_file(output_dir: Path) -> Path:
    path = output_dir / '周期(s).out'
    if path.exists():
        return path
    matches = [p for p in output_dir.glob('*.out') if p.name.endswith('(s).out')]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f'无法找到周期文件: {output_dir}')


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return default
    return [item.strip() for item in value.split(',') if item.strip()]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y'}


if __name__ == "__main__":
    import numpy as np
    from MRFcore.MRF import MRF
    from MRFcore.DataProcessing import DataProcessing
    from MRFcore.FragilityAnalysis import get_y

    smoke = os.environ.get('OPENSAS_IDA_SMOKE') == '1'
    temperature = _env_list('OPENSAS_IDA_TEMPS', ['-20'] if smoke else ['-20', '0', '20', '40'])
    motion_list = _env_list('OPENSAS_IDA_MOTIONS', ['1'] if smoke else [f'{i}' for i in range(1, 45)])
    max_ana = _env_int('OPENSAS_IDA_MAX_ANA', 1 if smoke else 80)
    parallel = _env_int('OPENSAS_IDA_PARALLEL', 1 if smoke else 8)
    trace_collapse = _env_bool('OPENSAS_IDA_TRACE_COLLAPSE', not smoke)
    output_root = os.environ.get('OPENSAS_IDA_OUTPUT_ROOT')
    # temperature = ['-20','-10','0','10','20','30','40']
    # temperature = ['40']

    
    for temp in temperature:

        model_name = f'MC8_SMABF_{temp}'
        if output_root:
            output_folder = Path(output_root) / model_name / 'MC8_IDA_data'
        elif smoke:
            output_folder = Path('Output_data') / '_codex_example_ida_smoke' / model_name / 'MC8_IDA_data'
        else:
            output_folder = Path('Output_data') / model_name / 'MC8_IDA_data'

        T1 = np.loadtxt(_period_file(Path('Output_data') / model_name / 'MC8_PO_out'))[0]
        # T1 = 2
        T, Sa = np.loadtxt(Path(r'Spectrum\MCE Level Spectrum.txt'), unpack=True)
        Sa_MCE = get_y(T, Sa, T1)


        # 1. Perform IDA
        note = 'IDA of a Eight-story steel moment resisting frame'
        model = MRF(model_name,Nstory=8 ,Nbay=3,heights=[5500, 4300, 4300, 4300, 4300, 4300, 4300, 4300], notes=note, script='py')
        # motion_list = ['1', '7', '13', '19', '25', '30', '37', '43' , '28', '34', '17']
        # motion_list = ['1', '2']
        model.select_ground_motions(motion_list, suffix='.txt')
        model.set_running_parameters(
            Output_dir=output_folder,
            fv_duration=0 if smoke else 30,
            display=False,
            auto_quit=True,
            folder_exists='delete' if smoke else 'overwrite'
        )
        model.run_IDA(
            T1,
            0.2 * Sa_MCE,
            0.2 * Sa_MCE,
            0.02,
            max_ana=max_ana,
            parallel=parallel,
            print_result=False,
            trace_collapse=trace_collapse,
            resume=True
        )


        # 2. Read results
        time0 = time.time()
        model = DataProcessing(output_folder)
        model.set_output_dir(output_folder.parent / (output_folder.name+'_out'), cover=1)
        model.read_results('mode', 'IDR', 'CIDR', 'PFA', 'PFV', 'shear', 'panelZone', 'beamHinge', 'columnHinge', print_result=True)
        time1 = time.time()
        print('耗时', time1 - time0)


        # # 3. Fragility analysis
        # model = FragilityAnalysis(
        #     Path('Output_data') / model_name / 'MC8_IDA_data_out',
        #     EDP_types=['IDR', 'RIDR','PFA'],
        #     collapse_limit=0.1,
        # )
        # model.calc_IDA('IDR')
        # model.calc_IDA('RIDR')
        # model.calc_IDA('PFA')
        # model.frag_curve('IDR', {'DS-1': 0.02, 'DS-2': 0.03,'DS-3': 0.05}, beta = ['betaD'],IM_limit=2.0)
        # model.frag_curve('RIDR',{'DS-1': 0.002, 'DS-2': 0.005, 'DS-3': 0.01},beta = ['betaD'],IM_limit=2.0)
        # model.frag_curve('PFA', {'DS-1': 0.5,   'DS-2': 1,   'DS-3': 1.5},beta = ['betaD'])

        # model.exceedance_probability('IDR', {'DS-1': 0.05, 'DS-2': 0.1},beta=0.5)
        # model.exceedance_probability('RIDR', {'DS-1': 0.002, 'DS-2': 0.005,'DS-3':0.01},beta=0.5)

        # hazard_curves = np.loadtxt('data\\Hzarad_curves_LA_Soil-D\\1.5.txt')
        # model.manual_probability('IDR', (0.001, 2), (0.001, 0.1), hazard_curves,fragility_type='computed')
        # model.manual_probability('RIDR', (0.001, 2), (0.0001, 0.02), hazard_curves,fragility_type='computed')

        # miuT = np.loadtxt(Path('Output_data') / model_name / 'MC8_PO_out' / '延性系数.txt')

        # miuT = 2.5
        # model.collapse_evaluation(T=T1, MCE_spec=Path(r'Spectrum\MCE Level Spectrum.txt'), SF_spec=1,miuT=miuT)

        # model.visualization()
        # model.save_data(Path('Output_data') / model_name / 'MC8_IDA_data_frag')
    print('All done')
