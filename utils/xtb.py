import subprocess
from pathlib import Path
import os

xtb_path = Path(os.getcwd()) / 'utils/xtb/bin/xtb'

class XTB():
    def __init__(self, process_path='./'):
        self.process_path = process_path

    def run_process(self, command):
        if isinstance(command, str): command = command.split()
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=self.process_path)
        return process.stdout.decode('UTF-8')

    def process_spin(self, mol_path: str, charge):
        command = f'{xtb_path} {mol_path} --sp --chrg {charge}' # generating molecule spin with xtb
        return self.run_process(command)

    def process_geometry_optimization(self, mol_path: str, charge):
        command = f'{xtb_path} {mol_path} --chrg {charge} --opt verytight' # optimizing molecule with xtb
        return self.run_process(command)

    def process_hessian_matrix(self, optimized_mol_path, charge):
        command = f'{xtb_path} {str(optimized_mol_path)} --hess --chrg {charge}' # optimizing molecule with xtb
        return self.run_process(command)

    def delete_xtb_files(self):
        files = [
            'wbo',
            'xtbopt.log',
            'xtbrestart',
            'xtbtopo.mol',
            'charges',
            'xtblast.xyz',
            '.xtboptok',
            'NOT_CONVERGED',
            '.sccnotconverged',
            'vibspectrum',
            'g98.out',
            'hessian',
        ]

        for filename in files:
            file = self.process_path / Path(filename)
            if file.is_file(): file.unlink()

