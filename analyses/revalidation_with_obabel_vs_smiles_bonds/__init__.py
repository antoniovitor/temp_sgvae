import shutil
from pathlib import Path
from tqdm import tqdm
from . import obabel_vs_smiles_validation
import pandas as pd

def validate(directories = None):
    base_path = Path('datasource/all_sampled_ions')

    default_drectories = [
        # # Partial optimization
        # base_path / 'calc01_popt_kl',
        # base_path / 'calc01_popt_prop',
        # base_path / 'calc01_popt_standard',
        # base_path / 'calc02_popt',
        # base_path / 'calc03_popt',
        # base_path / 'calc04_popt',

        # Final optimization
        base_path / 'calc01_fopt_kl',
        base_path / 'calc01_fopt_prop',
        base_path / 'calc01_fopt_standard',
        base_path / 'calc02_fopt',
        base_path / 'calc03_fopt',
        base_path / 'calc04_fopt',
    ]

    if not directories: directories = default_drectories

    analysis_base_path = Path('results/ions_validated_with_obabel_vs_smiles_bonds')
    analysis_base_path.mkdir(parents=True, exist_ok=True)
    tqdm_disable = True

    for dir in tqdm(directories, desc='Directories', disable=tqdm_disable):
        set_name = dir.name
        print(f'Validating {set_name}:')
        for subset in tqdm(['anions', 'cations'], leave=False, desc='Subsets', disable=tqdm_disable):
            subset_dir = dir / subset

            # creating dirs
            valid_dir = analysis_base_path / set_name / subset / 'valid'
            valid_dir.mkdir(parents=True, exist_ok=True)
            invalid_dir = analysis_base_path / set_name / subset / 'invalid'
            invalid_dir.mkdir(parents=True, exist_ok=True)

            
            valid_smiles = []
            invalid_smiles = []
            for xyz_path in tqdm(list(subset_dir.iterdir()), leave=False, desc='smiles', disable=tqdm_disable):
                smiles = xyz_path.stem

                is_valid = obabel_vs_smiles_validation.validate(smiles, xyz_path)

                if(is_valid):
                    valid_smiles.append(smiles)
                    shutil.copy(xyz_path, valid_dir / f'{smiles}.xyz')
                else:
                    invalid_smiles.append(smiles)
                    shutil.copy(xyz_path, invalid_dir / f'{smiles}.xyz')
            
            valid_count = len(valid_smiles)
            invalid_count = len(invalid_smiles)
            total = valid_count + invalid_count
            print(f'\t subset {subset}: {valid_count}/{total} ({valid_count/total:.2%})')
            # print(f'\t\t Valid smiles: {json.dumps(random.sample(valid_smiles, min(10, len(valid_smiles))))}')
            # print(f'\t\t Invalid smiles: {json.dumps(random.sample(invalid_smiles, min(10, len(invalid_smiles))))}')

def filter_valid_ions():
    source_path = Path('datasource/sampled_ions')
    dest_path = Path('datasource/sampled_ions_revalidated')
    valid_ions_path = Path('results/ions_validated_with_obabel_vs_smiles_bonds/calc04_fopt')

    for subset in ['anions', 'cations']:
        subset_dest_path = dest_path / subset
        subset_dest_path.mkdir(parents=True, exist_ok=True)

        smiles_df = pd.read_csv(source_path / subset / 'novel_molecules.csv')
        prop_df = pd.read_csv(source_path / subset / f'{subset}_with_prop.csv')

        # getting valid ions
        subset_valid_path = valid_ions_path / subset / 'valid'
        valid_smiles = [path.stem for path in list(subset_valid_path.glob('*'))]

        valid_smiles_df = smiles_df[smiles_df['smiles'].isin(valid_smiles)]

        # print(valid_smiles_df[valid_smiles_df['smiles'].duplicated()])
        print(valid_smiles_df.drop_duplicates(subset='smiles', keep='last'))



