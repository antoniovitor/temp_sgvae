from openbabel import pybel
from utils import xyz
import numpy as np

def _get_bond_matrix(mol):
    atoms_len = mol.NumAtoms()

    bonds_matrix = np.zeros((atoms_len, atoms_len), dtype=int)
    for bond in pybel.ob.OBMolBondIter(mol):
        idx1, idx2  = bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1
        bonds_matrix[idx1, idx2] = 1
        bonds_matrix[idx2, idx1] = 1

    return bonds_matrix

def validate(smiles, xyz_path):
    smiles_molecule = pybel.readstring('smi', smiles)
    smiles_molecule.addh()
    smiles_mol = smiles_molecule.OBMol

    xyz_mol = xyz.xyz2pybel(xyz_path)

    smiles_matrix = _get_bond_matrix(smiles_mol)
    xyz_matrix = _get_bond_matrix(xyz_mol)

    return np.array_equal(smiles_matrix, xyz_matrix)    
