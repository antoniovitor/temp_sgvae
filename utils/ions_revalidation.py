from pathlib import Path
from utils import bonds_length_validation
import shutil
import json
import random
import sys

base_path = Path('datasource/generated_ions')

default_drectories = [
    base_path / 'calc01_fopt_kl',
    base_path / 'calc01_fopt_prop',
    base_path / 'calc01_fopt_standard',
    base_path / 'calc02_fopt',
    base_path / 'calc03_fopt',
    base_path / 'calc04_fopt',
]

analysis_base_path = Path('results/ions_validated')
analysis_base_path.mkdir(parents=True, exist_ok=True)

def validate_ions(directories = default_drectories):
    file = open('validation.log', 'w')
    sys.stdout = file

    for dir in directories:
        set_name = dir.name
        print(f'Validating {set_name}:')
        for subset in ['anions', 'cations']:
            subset_dir = dir / subset
            valid_counts = 0
            total_counts = 0
            valid_smiles = []
            invalid_smiles = []
            for xyz_path in subset_dir.iterdir():
                smiles = xyz_path.stem
                total_counts += 1

                is_valid = bonds_length_validation.validate_bonds_lengths(smiles, xyz_path, subset)

                if(is_valid):
                    valid_counts += 1
                    save_dir = analysis_base_path / set_name / subset
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f'{smiles}.xyz'
                    # shutil.copy(xyz_path, save_path)
                    valid_smiles.append(smiles)
                else:
                    invalid_smiles.append(smiles)
            print(f'\t subset {subset}: {valid_counts}/{total_counts} ({valid_counts/total_counts:.2%})')
            print(f'\t\t Valid smiles: {json.dumps(random.sample(valid_smiles, min(10, len(valid_smiles))))}')
            print(f'\t\t Invalid smiles: {json.dumps(random.sample(invalid_smiles, min(10, len(invalid_smiles))))}')
