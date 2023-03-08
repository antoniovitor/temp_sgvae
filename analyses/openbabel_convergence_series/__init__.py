from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from utils import babel
from tqdm import tqdm
from utils.rmsd import compare_geometries
from .plots import plot_charts
from openbabel import pybel

######### GLOBAL VARIABLES
seed = 42

dataset_path = Path('datasource/original/dataset_complete/')
analysis_path = Path('results/openbabel_convergence_series')
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

def _process_smiles(smiles_df):
    data = []
    for _, row in tqdm(smiles_df.iterrows(), desc='Processing smiles', total=len(smiles_df.index)):
        mol_id, smiles, subset = row[['mol_id', 'smiles', 'subset']]

        smiles_folder = xyz_path / smiles # paths
        smiles_folder.mkdir(parents=True, exist_ok=True)
        dft_path = smiles_folder / 'dft.xyz'
        obabel_path = smiles_folder / f'force_field.xyz'

        source_path = dataset_path / f'{subset}_xyz/{mol_id}.xyz'
        shutil.copy(source_path, smiles_folder / f'dft.xyz') # moving xyz file with dft geometry

        for ff_id in tqdm(range(10), desc=f'Processing {mol_id} xyz', leave=False):
            mol = pybel.readstring('smi', smiles)
            mol.addh()
            mol.make3D(steps=0)

            rmsd_list = []
            for step in range(1000):
                mol.localopt(steps=1)
                mol.write('xyz', str(obabel_path), overwrite=True)
                rmsd = compare_geometries(dft_path, obabel_path)
                rmsd_list.append(rmsd)
            
            data.append({
                'mol_id': mol_id,
                'ff_id': ff_id,
                'smiles': smiles,
                'rmsd_list': rmsd_list,
                'mean': np.mean(rmsd_list),
                'std': np.std(rmsd_list),
            })

    df = pd.DataFrame(data)
    df.to_json(analysis_path / 'df.json')
    df.drop('rmsd_list', axis=1).to_csv(analysis_path / 'df.csv')

def analyze():
    dft_df_path = analysis_path / 'dft_diff.json'
    if(dft_df_path.exists()):
        plot_charts()
        return

    # sampling from anions and cations
    anions_df = pd.read_csv(dataset_path / 'anions.csv').sample(n=5, random_state=seed)
    anions_df['subset'] = 'anions'
    cations_df = pd.read_csv(dataset_path / 'cations.csv').sample(n=5, random_state=seed)
    cations_df['subset'] = 'cations'
    df = pd.concat([anions_df, cations_df])

    if(not xyz_path.exists()): # check if xyz was generated
        xyz_path.mkdir(parents=True, exist_ok=True)
        _process_smiles(df)

    plot_charts()