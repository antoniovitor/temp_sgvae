from rdkit import Chem
from utils.xtb import XTB
import utils.babel as babel
from pathlib import Path
import shutil
import os

########## VALIDATION
def check_magnetic_moment(logs: str):
    spin = -1
    for line in logs.split('\n'):
        if 'spin' in line:
            spin = float(line.split()[2])
            break
    
    return spin == 0

def check_geometry_convergence(logs: str):
    search_str = 'GEOMETRY OPTIMIZATION CONVERGED'
    converged = [line for line in logs.split('\n') if search_str in line]
    return len(converged) > 0

def check_distances(logs: str):
    lines = logs.split('\n')
    start = -1
    for i, line in enumerate(lines):
        if('selected distances' in line):
            start = i + 3
            count = int(line.split()[1])

    data = [float(item.split()[6]) for item in lines[start:start + count]]

    if max(data) < 3:
        return True
    return False

def check_hessian_matrix(logs: str):
    search_str = 'projected vibrational frequencies'
    lines = logs.split('\n')
    start = [i for i, line in enumerate(lines) if search_str in line][0] + 1

    i = start
    while('eigval' in lines[i]):
        eigenvalues = [float(value) for value in lines[i].split()[2:]]
        if(min(eigenvalues) < 0):
            return False
        i += 1

    return True

# TODO: filter smiles already generated when sampling
def validate_smiles(smiles: str, subset: str, existing_smiles, save_path=None):
    if not smiles:
        return { 'result': False, 'error': 'SMILE_DOES_NOT_EXIST' }

    mol = Chem.MolFromSmiles(smiles)

    # check if RDkit can read the smile
    if not mol:
        return { 'result': False, 'error': 'RDKIT_CAN_NOT_READ_SMILE' }

    # check if can recreate smile from mol object
    canonical_smile = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    if not canonical_smile:
        return { 'result': False, 'error': 'CAN_NOT_RECREATE_SMILE_FROM_MOL_OBJECT' }
    
    # check if canonical smile already exist and isn't in dataset
    if canonical_smile in existing_smiles:
        return { 'result': False, 'error': 'SMILE_ALREADY_EXISTS_IN_DATASET' }

    # check if smile is monovalent
    if subset == 'anions' and Chem.rdmolops.GetFormalCharge(mol) != -1: 
        return { 'result': False, 'error': 'FORMAL_CHARGE_OF_ANION_IS_NOT_-1' }
    elif subset == 'cations' and Chem.rdmolops.GetFormalCharge(mol) != 1:
        return { 'result': False, 'error': 'FORMAL_CHARGE_OF_CATION_IS_NOT_+1' }

    # check if smile presents charges
    if subset == 'anions' and '-' not in canonical_smile: 
        return { 'result': False, 'error': 'IONS_DOES_NOT_HAVE_FORMAL_CHARGE' }
    elif subset == 'cations' and '+' not in canonical_smile:
        return { 'result': False, 'error': 'IONS_DOES_NOT_HAVE_FORMAL_CHARGE' }

    # creates or clean temp directory
    temp_dir = Path('temp/')
    if temp_dir.exists():  shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    ## geometry validation
    # checking OpenBabel geometry optimization
    openbabel_filename = 'openbabel_mol.xyz'
    mol = babel.create_mol_file_from_smile(canonical_smile, temp_dir / openbabel_filename)
    if not mol:
        return { 'result': False, 'error': 'OPENBABEL_CAN_NOT_OPTIMIZE_SMILE' }


    # checking XTB spin validation
    charge =  -1 if subset == 'anions' else 1
    xtb = XTB(process_path=temp_dir)
    spin_log = xtb.process_spin(openbabel_filename, charge) # checking magnetic moment
    if not check_magnetic_moment(spin_log):
        return { 'result': False, 'error': 'SPIN_IS_NOT_ZERO' }

    # checking XTB geometry optimization
    optimization_log = xtb.process_geometry_optimization(openbabel_filename, charge)
    if not check_geometry_convergence(optimization_log):
        return { 'result': False, 'error': 'GEOMETRY_DID_NOT_CONVERGE' }
    if not check_distances(optimization_log):
        return { 'result': False, 'error': 'BONDS_HAVE_MORE_THAN_3_ANGSTROMS' }

    # checking hessian matrix
    opt_filename = 'xtbopt.xyz'
    hessian_log = xtb.process_hessian_matrix(opt_filename, charge) # TODO: include path of optimized molecule
    if not check_hessian_matrix(hessian_log):
        return { 'result': False, 'error': 'HESSIAN_MATRIX_HAVE_NEGATIVES_EIGENVALUES' }

    # saving files
    with open(save_path / f'{smiles}.spin.log', 'w') as file: # saving spin file
        file.write(spin_log)
    with open(save_path / f'{smiles}.opt.log', 'w') as file: # saving optimization file
        file.write(optimization_log)
    with open(save_path / f'{smiles}.hessian.log', 'w') as file: # saving hessian matrix file
        file.write(hessian_log)
    os.replace(temp_dir / 'xtbopt.xyz', save_path / f'{smiles}.xyz') # saving geometry file

    # delete temp directory
    shutil.rmtree(temp_dir)

    return { 'result': True, 'smile': canonical_smile }