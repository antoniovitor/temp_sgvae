from openbabel import pybel
import numpy as np
import pandas as pd
from utils.bonds_length_validation.covalent_radii import covalent_radii_dict
from utils import xyz
from utils.xyz import read_xyz

covalent_radius_coefficients = np.array([.8, 1.2])

minimux_count = 10
std_factor = 5

boundaries = dict()
selected_analysis = 'bonds_from_smiles'
# selected_analysis = 'bonds_from_openbabel_GetLength'
# selected_analysis = 'bonds_from_openbabel_GetEquibLength'
for subset in ['anions', 'cations']:
    df = pd.read_json(f'results/bonds_analysis/{subset}/{selected_analysis}/bonds_distribution.json')
    # df = df.set_index('bond_id')

    # Case 1
    df['boundary_min'] = df['mean'] - df['std'] * std_factor # defining boundaries
    df['boundary_max'] = df['mean'] + df['std'] * std_factor

    # # Case 2
    # df['boundary_min'] = df['min'] # defining boundaries
    # df['boundary_max'] = df['max']

    # # Case 3
    # df['boundary_min'] = df['mean'] - df['std'] * std_factor # defining boundaries
    # df['boundary_max'] = df['mean'] + df['std'] * std_factor
    
    # print(df)
    # df.loc['C-H (1)']
    # print(df.loc['C-H (1)'])

    # df = df.loc[df['count'] >= minimux_count] # filtering

    subset_dict = dict()
    for i, row in df.iterrows():
        subset_dict[row['bond_id']] = [row['boundary_min'], row['boundary_max']]
    boundaries[subset] = subset_dict


########## FUNCTIONS
def get_bond_boundary(bond_id, subset):
    return boundaries[subset].get(bond_id)

def calc_covalent_radius(atom1, atom2):
    return covalent_radii_dict[atom1] + covalent_radii_dict[atom2]

def check_bond_limits(atom1, atom2, order, length, subset):
    descriptor = '-'.join(sorted([atom1, atom2])) # Ex: 'C-H' (alphabetical order)
    index = f'{descriptor} ({order})'

    # check bond length using empirical limits
    bond_boundary = get_bond_boundary(index, subset)
    if bond_boundary:
        length_min, length_max = bond_boundary
        if (not length_min <= length <= length_max):
            print(f'Error at bond {index}: length {length}')
        return length_min <= length <= length_max
    
    print(f'Using covalent radius for {index} in subset {subset}')

    # check bond length using covalent radius
    covalent_bond_radius = calc_covalent_radius(atom1, atom2)
    length_min, length_max = covalent_radius_coefficients * covalent_bond_radius
    return length_min <= length <= length_max

def validate_bonds_lengths(smiles, xyz_path, subset):
    mol = pybel.readstring('smi', smiles)
    mol.addh()
    xyz_positions = read_xyz(xyz_path)['positions']
    
    for bond in pybel.ob.OBMolBondIter(mol.OBMol):
        idx1, idx2  = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom1, atom2 = xyz_positions[idx1 - 1], xyz_positions[idx2 - 1]
        position1, position2 = atom1['position'], atom2['position']
        elem1, elem2 = atom1['element'], atom2['element']

        length = np.linalg.norm(position1 - position2)
        order = bond.GetBondOrder()

        descriptor = '-'.join(sorted([elem1, elem2])) 

        valid_bond = check_bond_limits(elem1, elem2, order, length, subset)
        if not valid_bond:
            # print(f'smiles: {smiles}, bond {descriptor} broken')
            # print(f'{atom1}: {position1}, {atom2}: {position2}, order: {order}, length: {length}')
            return False
    return True