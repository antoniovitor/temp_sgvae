import pandas as pd
from openbabel import pybel
import numpy as np
from pathlib import Path
import args_parser
from analyses.bond_lengths import plots
from tqdm import tqdm
from utils.xyz import read_xyz

########## GLOBAL VARIABLES
args = args_parser.get_args()

analysis_base_path = Path('results/bonds_analysis')
analysis_base_path.mkdir(parents=True, exist_ok=True)

xyz_base_path = Path(f'datasource/original/dataset_complete/')

########## FUNCTIONS
def get_bond_id(element1, element2, order): # Ex: 'C-H (1)' (alphabetical order)
    return get_bond_descriptor(element1, element2) + f' ({order})'

def get_bond_descriptor(element1, element2):  # Ex: 'C-H' (alphabetical order)
    return '-'.join(sorted([element1, element2]))

def analyze_with_bonds_from_smiles(dataset: pd.DataFrame, save_path):
    save_path.mkdir(parents=True, exist_ok=True)

    mols_data = []
    desc = 'bonds from SMILES'
    for i, row in tqdm(dataset.iterrows(), total=(len(dataset.index)), desc=desc):
        mol_id, smiles, subset = row[['mol_id', 'smiles', 'subset']]
        atoms_positions = read_xyz(xyz_base_path / f'{subset}_xyz/{mol_id}.xyz')['positions']
        mol = pybel.readstring('smi', smiles)
        mol.addh()

        for bond in pybel.ob.OBMolBondIter(mol.OBMol):
            idx1, idx2  = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            atom1, atom2 = atoms_positions[idx1 - 1], atoms_positions[idx2 - 1]
            position1, position2 = atom1['position'], atom2['position']
            elem1, elem2 = atom1['element'], atom2['element']

            order = bond.GetBondOrder()
            length = np.linalg.norm(position1 - position2)

            mols_data.append({
                'mol_id': mol_id,
                'bond_id': get_bond_id(elem1, elem2, order),
                'bond': get_bond_descriptor(elem1, elem2),
                'order': order,
                'length': length,
            })

    mols_df = pd.DataFrame(mols_data) # creating dataframes
    mols_df.set_index(['mol_id', 'bond_id'])

    bonds_df = (
        mols_df
        .groupby(['bond_id', 'bond', 'order'], as_index=False)['length']
        .agg(['mean', 'std', 'count', 'min', 'max'])
        .reset_index()
    )
    
    mols_df.to_json(save_path / 'mols_bonds.json')
    mols_df.to_csv(save_path / 'mols_bonds.csv')

    bonds_df.to_json(save_path / 'bonds_distribution.json')
    bonds_df.to_csv(save_path / 'bonds_distribution.csv')

def analyze_with_bonds_from_openbabel_GetLength(dataset: pd.DataFrame, save_path):
    save_path.mkdir(parents=True, exist_ok=True)

    mols_data = []
    desc = 'bonds from OpenBabel using GetLength'
    for i, row in tqdm(dataset.iterrows(), total=(len(dataset.index)), desc=desc):
        mol_id, subset = row[['mol_id', 'subset']]
        
        # converting XYZ file to mol object
        xyz_path = xyz_base_path / f'{subset}_xyz/{mol_id}.xyz'
        xyz_file = open(xyz_path).read()
        
        obconversion = pybel.ob.OBConversion()
        obconversion.SetInFormat('xyz')
        mol = pybel.ob.OBMol()
        obconversion.ReadString(mol, xyz_file)
        

        for bond in pybel.ob.OBMolBondIter(mol):
            idx1, idx2  = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            elem1 = pybel.ob.GetSymbol(mol.GetAtom(idx1).GetAtomicNum())
            elem2 = pybel.ob.GetSymbol(mol.GetAtom(idx2).GetAtomicNum())

            order = bond.GetBondOrder()
            length = bond.GetLength()

            mols_data.append({
                'mol_id': mol_id,
                'bond_id': get_bond_id(elem1, elem2, order),
                'bond': get_bond_descriptor(elem1, elem2),
                'order': order,
                'length': length,
            })

    mols_df = pd.DataFrame(mols_data) # creating dataframes
    mols_df.set_index(['mol_id', 'bond_id'])

    bonds_df = (
        mols_df
        .groupby(['bond_id', 'bond', 'order'], as_index=False)['length']
        .agg(['mean', 'std', 'count', 'min', 'max'])
        .reset_index()
    )
    
    mols_df.to_json(save_path / 'mols_bonds.json')
    mols_df.to_csv(save_path / 'mols_bonds.csv')

    bonds_df.to_json(save_path / 'bonds_distribution.json')
    bonds_df.to_csv(save_path / 'bonds_distribution.csv')

def analyze_with_bonds_from_openbabel_GetEquibLength(dataset: pd.DataFrame, save_path):
    save_path.mkdir(parents=True, exist_ok=True)

    mols_data = []
    desc = 'bonds from OpenBabel using GetEquibLength'
    for i, row in tqdm(dataset.iterrows(), total=(len(dataset.index)), desc=desc):
        mol_id, subset = row[['mol_id', 'subset']]
        
        # converting XYZ file to mol object
        xyz_path = xyz_base_path / f'{subset}_xyz/{mol_id}.xyz'
        xyz_file = open(xyz_path).read()
        
        obconversion = pybel.ob.OBConversion()
        obconversion.SetInFormat('xyz')
        mol = pybel.ob.OBMol()
        obconversion.ReadString(mol, xyz_file)
        

        for bond in pybel.ob.OBMolBondIter(mol):
            idx1, idx2  = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            elem1 = pybel.ob.GetSymbol(mol.GetAtom(idx1).GetAtomicNum())
            elem2 = pybel.ob.GetSymbol(mol.GetAtom(idx2).GetAtomicNum())

            order = bond.GetBondOrder()
            length = bond.GetEquibLength()

            mols_data.append({
                'mol_id': mol_id,
                'bond_id': get_bond_id(elem1, elem2, order),
                'bond': get_bond_descriptor(elem1, elem2),
                'order': order,
                'length': length,
            })

    mols_df = pd.DataFrame(mols_data) # creating dataframes
    mols_df.set_index(['mol_id', 'bond_id'])

    bonds_df = (
        mols_df
        .groupby(['bond_id', 'bond', 'order'])['length']
        .agg(['mean', 'std', 'count', 'min', 'max'])
        .reset_index()
    )
    
    mols_df.to_json(save_path / 'mols_bonds.json')
    mols_df.to_csv(save_path / 'mols_bonds.csv')

    bonds_df.to_json(save_path / 'bonds_distribution.json')
    bonds_df.to_csv(save_path / 'bonds_distribution.csv')

########## ANALYSIS
def run_bond_length_analysis():
    for subset in ['anions', 'cations']:
        analysis_path  = analysis_base_path / subset
        # if(analysis_path.exists()): continue

        print(f'Analyzing {subset} dataset...')
        df = pd.read_csv(f'datasource/original/dataset_complete/{subset}.csv')
        df['subset'] = subset

        bonds_from_smiles_path = analysis_path / 'bonds_from_smiles'
        analyze_with_bonds_from_smiles(df, bonds_from_smiles_path)
        
        bonds_from_openbabel_GetLength_path = analysis_path / 'bonds_from_openbabel_GetLength'
        analyze_with_bonds_from_openbabel_GetLength(df, bonds_from_openbabel_GetLength_path)
        
        bonds_from_openbabel_GetEquibLength_path = analysis_path / 'bonds_from_openbabel_GetEquibLength'
        analyze_with_bonds_from_openbabel_GetEquibLength(df, bonds_from_openbabel_GetEquibLength_path)


    ######### PLOTTING
    # plots.plot_bonds_charts()