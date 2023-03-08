from openbabel import pybel

def create_mol_file_from_smile(smile, path, steps=None):
    mol = pybel.readstring('smi', smile)
    mol.addh()
    mol.make3D(steps=0 if steps is not None else 50)
    mol.localopt(steps=steps)
    mol.write('xyz', str(path), overwrite=True)

    return mol