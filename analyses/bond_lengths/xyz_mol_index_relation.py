######### WARNING: this file is only for tests
from pathlib import Path
import pandas as pd
from openbabel import pybel
from tqdm import tqdm
import args_parser
import analyses.bond_lengths as bond_lengths

atomic_numbers = {
    'H': 1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Ar': 18,
    'K': 19,
    'Ca': 20,
    'Sc': 21,
    'Ti': 22,
    'V': 23,
    'Cr': 24,
    'Mn': 25,
    'Fe': 26,
    'Co': 27,
    'Ni': 28,
    'Cu': 29,
    'Zn': 30,
    'Ga': 31,
    'Ge': 32,
    'As': 33,
    'Se': 34,
    'Br': 35,
    'Kr': 36,
    'Rb': 37,
    'Sr': 38,
    'Y': 39,
    'Zr': 40,
    'Nb': 41,
    'Mo': 42,
    'Tc': 43,
    'Ru': 44,
    'Rh': 45,
    'Pd': 46,
    'Ag': 47,
    'Cd': 48,
    'In': 49,
    'Sn': 50,
    'Sb': 51,
    'Te': 52,
    'I': 53,
    'Xe': 54,
    'Cs': 55,
    'Ba': 56,
    'La': 57,
    'Ce': 58,
    'Pr': 59,
    'Nd': 60,
    'Pm': 61,
    'Sm': 62,
    'Eu': 63,
    'Gd': 64,
    'Tb': 65,
    'Dy': 66,
    'Ho': 67,
    'Er': 68,
    'Tm': 69,
    'Yb': 70,
    'Lu': 71,
    'Hf': 72,
    'Ta': 73,
    'W': 74,
    'Re': 75,
    'Os': 76,
    'Ir': 77,
    'Pt': 78,
    'Au': 79,
    'Hg': 80,
    'Tl': 81,
    'Pb': 82,
    'Bi': 83,
    'Po': 84,
    'At': 85,
    'Rn': 86,
    'Fr': 87,
    'Ra': 88,
    'Ac': 89,
    'Th': 90,
    'Pa': 91,
    'U': 92,
    'Np': 93,
    'Pu': 94,
    'Am': 95,
    'Cm': 96,
    'Bk': 97,
    'Cf': 98,
    'Es': 99,
    'Fm': 100,
    'Md': 101,
    'No': 102,
    'Lr': 103,
    'Rf': 104,
    'Db': 105,
    'Sg': 106,
    'Bh': 107,
    'Hs': 108,
    'Mt': 109,
    'Ds': 110,
    'Rg': 111,
    'Cn': 112,
    'Nh': 113,
    'Fl': 114,
    'Mc': 115,
    'Lv': 116,
    'Ts': 117,
    'Og': 118,
}

########## GLOBAL VARIABLES
args = args_parser.get_args()

bonds_analysis_path = Path('results/bonds_analysis')
bonds_analysis_path.mkdir(parents=True, exist_ok=True)

def check_xyz_mol_index_relation():
    anions = pd.read_csv(f'datasource/original/dataset_complete/anions.csv')
    anions['subset'] = 'anions'
    cations = pd.read_csv(f'datasource/original/dataset_complete/cations.csv')
    cations['subset'] = 'cations'

    dataset = pd.concat([anions, cations])
    xyz_base_path = Path(f'datasource/original/dataset_complete/')

    for i, row in tqdm(dataset.iterrows(), total=(len(dataset.index))):
        mol_id, smiles, subset = row[['mol_id', 'smiles', 'subset']]
        xyz = bond_lengths.read_xyz(xyz_base_path / f'{subset}_xyz/{mol_id}.xyz')

        mol = pybel.readstring('smi', smiles)
        mol.addh()

        for bond in pybel.ob.OBMolBondIter(mol.OBMol):
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()

            atom_obabel_num1 = mol.atoms[atom1_idx - 1].atomicnum
            atom_obabel_num2 = mol.atoms[atom2_idx - 1].atomicnum

            atomic_xyz_num1 = atomic_numbers[xyz[atom1_idx - 1]['element']]
            atomic_xyz_num2 = atomic_numbers[xyz[atom2_idx - 1]['element']]

            if(atomic_xyz_num1 != atom_obabel_num1):
                print(f'Differs!')
                print(f'mol_id: {mol_id}, smiles: {smiles}')
            if(atomic_xyz_num2 != atom_obabel_num2):
                print(f'Differs!')
                print(f'mol_id: {mol_id}, smiles: {smiles}')
