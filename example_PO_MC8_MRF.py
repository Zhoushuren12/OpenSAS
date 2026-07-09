import sys
from pathlib import Path
sys.path.append(Path.cwd().as_posix())
from MRFcore.MRF import MRF
from MRFcore.DataProcessing import DataProcessing
from MRFcore.QuakeReadPushover import QuakeReadPushover
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

    model_name = 'MC8_SMRF'
    output_folder = Path('Output_data') / model_name / 'MC8_PO'

    # 1. Perform pushover analysis
    note = 'pushover analysis of a four-story steel moment resisting frame'
    model = MRF(model_name,Nstory=8 ,Nbay=3,heights=[5500, 4300, 4300, 4300, 4300, 4300, 4300, 4300], notes=note, script='py')
    model.set_running_parameters(Output_dir=output_folder, display=True, auto_quit=False)
    model.run_pushover(print_result=True)
    QuakeReadPushover(output_folder)

    # 2. Read results
    model = DataProcessing(output_folder)
    model.set_output_dir(output_folder.parent / (output_folder.name+'_out'), cover=1)
    model.read_results('mode', 'IDR', 'CIDR', 'PFA', 'PFV', 'shear', 'panelZone', 'beamHinge', 'columnHinge', print_result=True)
    model.read_pushover(H=18400)

