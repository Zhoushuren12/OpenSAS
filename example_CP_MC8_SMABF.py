from pathlib import Path
from MRFcore.MRF import MRF
from MRFcore.DataProcessing import DataProcessing
from MRFcore.QuakeReadPushover import QuakeReadPushover


if __name__ == "__main__":
    
    model_name = f'MC8_SMABF'
    output_folder = Path('Output_data') / model_name / 'MC8_CP'
    
    # 1. Perform pushover analysis
    note = 'pushover analysis of a eight-story steel moment resisting frame'
    model = MRF(model_name,Nstory=8 ,Nbay=3,heights=[5500, 4300, 4300, 4300, 4300, 4300, 4300, 4300], notes=note, script='py')
    model.set_running_parameters(Output_dir=output_folder, display=True, auto_quit=False)
    cp_path = [0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06, 0.08, -0.08,0]
    model.run_cyclic_pushover(RDR_path=cp_path, print_result=True)

    # 2. Read results
    model = DataProcessing(output_folder)
    model.set_output_dir(output_folder.parent / (output_folder.name+'_out'), cover=1)
    model.read_results('mode', 'IDR', 'CIDR', 'PFA', 'PFV', 'shear', 'panelZone', 'beamHinge', 'columnHinge', print_result=True)
    model.read_cyclic_pushover(H=16300)

