import sys
from pathlib import Path
sys.path.append(Path.cwd().as_posix())
from MRFcore.MRF import MRF
from MRFcore.DataProcessing import DataProcessing
import random
import builtins
original_open = builtins.open
def utf8_open(*args, **kwargs):
    if 'encoding' not in kwargs:
        mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')
        if 'b' not in mode:  
            kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)
builtins.open = utf8_open

if __name__ == "__main__":


    Groundmove_level = ['CLE']
    # Groundmove_level = ['DBE','MCE']
    for level in Groundmove_level:

        output_folder = Path(f'Output_data\\MC8_MRF\\MC8_TH_{level}_data')
        note = 'time history of a four-story steel moment resisting frame'
        model = MRF('MC8_MRF',Nstory=8 ,Nbay=3,heights=[5500, 4300, 4300, 4300, 4300, 4300, 4300, 4300], notes=note, script='py')
        # model_list = [f'{i}' for i in [12,21,10,27,5,34,29,39,42,19,7]]
        # motion_list = [f'{i}' for i in range(1, 45) if i != 20 and i != 38]
        motion_list = [f'{i}' for i in range(1, 45)]
        model.select_ground_motions(motion_list, suffix='.txt')
        T1 = 3.636
        model.scale_ground_motions(method='c', para=(0.2*T1,1.5*T1),path_spec_code=Path(f'Spectrum\{level} Level Spectrum.txt')
                                ,save_SF=True,plot=False)
        model.set_running_parameters(Output_dir=output_folder, fv_duration=30, display=True, auto_quit=True,folder_exists='overwrite')
        model.run_time_history(print_result=False, parallel=11)


        # 2. Read results
        model = DataProcessing(output_folder)
        model.set_output_dir(output_folder.parent / (output_folder.name+'_out'), cover=1)
        model.read_results('mode', 'IDR', 'CIDR', 'PFA', 'PFV', 'shear', 'panelZone', 'beamHinge', 'columnHinge', print_result=True)
        model.read_th()
    print('All done!')

