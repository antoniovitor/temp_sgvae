import shutil
from pathlib import Path
from utils import smile_validation
from utils.xtb import XTB
from tqdm import tqdm

base_path = Path('datasource/generated_ions')

default_drectories = [
    # Partial optimization
    base_path / 'calc01_popt_kl',
    base_path / 'calc01_popt_prop',
    base_path / 'calc01_popt_standard',
    base_path / 'calc02_popt',
    base_path / 'calc03_popt',
    base_path / 'calc04_popt',

    # Final optimization
    base_path / 'calc01_fopt_kl',
    base_path / 'calc01_fopt_prop',
    base_path / 'calc01_fopt_standard',
    base_path / 'calc02_fopt',
    base_path / 'calc03_fopt',
    base_path / 'calc04_fopt',
]

analysis_base_path = Path('results/ions_validated_with_hessian_matrix')
analysis_base_path.mkdir(parents=True, exist_ok=True)
temp_dir = Path('temp/')


xtb = XTB(process_path=temp_dir)


def _validate_ions_with_hessian_matrix(xyz_path):
    hessian_logs = xtb.process_hessian_matrix(xyz_path.resolve())
    return smile_validation.check_hessian_matrix(hessian_logs)

def validate_ions_with_hessian_matrix(directories = default_drectories):
    for dir in tqdm(directories, desc='Directories'):
        set_name = dir.name
        print(f'Validating {set_name}:')
        for subset in tqdm(['anions', 'cations'][1:], leave=False, desc='Subsets'):
            subset_dir = dir / subset

            # creating dirs
            valid_dir = analysis_base_path / set_name / subset / 'valid'
            valid_dir.mkdir(parents=True, exist_ok=True)
            invalid_dir = analysis_base_path / set_name / subset / 'invalid'
            invalid_dir.mkdir(parents=True, exist_ok=True)

            
            valid_smiles = []
            invalid_smiles = []
            for xyz_path in tqdm(list(subset_dir.iterdir()), leave=False, desc='smiles'):
                smiles = xyz_path.stem

                charge =  -1 if subset == 'anions' else 1
                hessian_log = xtb.process_hessian_matrix(xyz_path.resolve(), charge)
                is_valid = smile_validation.check_hessian_matrix(hessian_log)
                if(is_valid):
                    with open(valid_dir / f'{smiles}.hessian.log', 'w') as file:
                        file.write(hessian_log)
                    valid_smiles.append(smiles)
                    shutil.copy(xyz_path, valid_dir / f'{smiles}.xyz')
                else:
                    with open(invalid_dir / f'{smiles}.hessian.log', 'w') as file:
                        file.write(hessian_log)
                    invalid_smiles.append(smiles)
                    shutil.copy(xyz_path, invalid_dir / f'{smiles}.xyz')
            
            valid_count = len(valid_smiles)
            invalid_count = len(invalid_smiles)
            total = valid_count + invalid_count
            print(f'\t subset {subset}: {valid_count}/{total} ({valid_count/total:.2%})')
            # print(f'\t\t Valid smiles: {json.dumps(random.sample(valid_smiles, min(10, len(valid_smiles))))}')
            # print(f'\t\t Invalid smiles: {json.dumps(random.sample(invalid_smiles, min(10, len(invalid_smiles))))}')
