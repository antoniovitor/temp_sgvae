from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from utils import babel
from tqdm import tqdm
from utils.rmsd import compare_geometries
from .plots import plot_charts

######### GLOBAL VARIABLES
seed = 42

dataset_path = Path('datasource/original/dataset_complete/')
analysis_path = Path('results/openbabel_optimization')
analysis_path.mkdir(parents=True, exist_ok=True)
xyz_path = analysis_path / 'xyz'

######### FUNCTIONS
def _compare_openbabel_vs_dft_optmization(smiles):
    smiles_folder = xyz_path / smiles
    dft_path = smiles_folder / 'dft.xyz'

    rmsd_list = []
    for i in tqdm(range(1000), desc='DFT vs OpenBabel geometries', leave=False):
        obabel_path = smiles_folder / f'{i}.xyz'
        rmsd = compare_geometries(dft_path, obabel_path)
        rmsd_list.append(rmsd)
    return rmsd_list

def _compare_openbabel_vs_openbabel_optimization(smiles):
    smiles_folder = xyz_path / smiles

    rmsd_list = []
    for i in tqdm(range(1000), desc='OpenBabel geometries', leave=False):
        for j in range(i + 1, 1000):
            obabel_path_1 = smiles_folder / f'{i}.xyz'
            obabel_path_2 = smiles_folder / f'{j}.xyz'
            rmsd = compare_geometries(obabel_path_1, obabel_path_2)
            rmsd_list.append(rmsd)
    return rmsd_list

def _move_xyz_file_from_dft(df):
    for i, row in df.iterrows():
        mol_id, smiles, subset = row[['mol_id', 'smiles', 'subset']]

        # moving xyz file with dft geometry
        source_path = dataset_path / f'{subset}_xyz/{mol_id}.xyz'
        smiles_folder = xyz_path / smiles
        smiles_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, smiles_folder / f'dft.xyz')

def _generate_openbabel_geometry(df):
    for i, row in tqdm(df.iterrows(), desc='Processing smiles', total=len(df.index)):
        mol_id, smiles = row[['mol_id', 'smiles']]

        # create xyz using openbabel force-field optimization
        smiles_folder = xyz_path / smiles
        smiles_folder.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(1000), desc=f'Processing {mol_id} xyz', leave=False):
            babel.create_mol_file_from_smile(smiles, smiles_folder / f'{i}.xyz')

def analyze_openbabel_optimization():
    dft_df_path = analysis_path / 'dft_diff.json'
    obabel_df_path = analysis_path / 'obabel_diff.json'
    if(obabel_df_path.exists()):
        plot_charts()
        return

    # sampling from anions and cations
    anions_df = pd.read_csv(dataset_path / 'anions.csv').sample(n=5, random_state=seed)
    anions_df['subset'] = 'anions'
    cations_df = pd.read_csv(dataset_path / 'cations.csv').sample(n=5, random_state=seed)
    cations_df['subset'] = 'cations'
    df = pd.concat([anions_df, cations_df])

    if(not xyz_path.exists()): # check if xyz was generated
        _move_xyz_file_from_dft(df)
        _generate_openbabel_geometry(df)

    dft_errors = []
    obabel_errors = []
    for smiles in tqdm(df['smiles'], desc='SMILES'):
        dft_rmsd = _compare_openbabel_vs_dft_optmization(smiles)
        dft_errors.append({
            'smiles': smiles,
            'rmsd': dft_rmsd,
            'mean': np.mean(dft_rmsd),
            'std': np.std(dft_rmsd),
        })
        obabel_rmsd = _compare_openbabel_vs_openbabel_optimization(smiles)
        obabel_errors.append({
            'smiles': smiles,
            'rmsd': obabel_rmsd,
            'mean': np.mean(obabel_rmsd),
            'std': np.std(obabel_rmsd),
        })
    dft_df = pd.DataFrame(dft_errors)
    obabel_df = pd.DataFrame(obabel_errors)

    dft_df.to_json(dft_df_path)
    obabel_df.to_json(obabel_df_path)
    

    dft_df.drop('rmsd', axis=1).to_csv(analysis_path / 'dft_diff.csv')
    obabel_df.drop('rmsd', axis=1).to_csv(analysis_path / 'obabel_diff.csv')

    plot_charts()