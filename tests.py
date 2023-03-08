######### WARNING: this file is only for tests
from pathlib import Path
import pandas as pd
from openbabel import pybel
import numpy as np
from tqdm import tqdm
import args_parser
import matplotlib.pyplot as plt
import bootstrap
bootstrap.bootstrap()


# openbabel_optimization.analyze_openbabel_optimization()

# bond_lengths.analyze_bond_length_distribution()

# analysis_base_path = Path('results/bonds_analysis')
# bond_lengths.plots.plot_bonds_charts(analysis_base_path / 'anions')

# anions_df = pd.read_json(f'utils/bonds_length_validation/anions_boundaries.json')[['identifier', 'length_boundaries', 'count']]
# anions_df = anions_df.loc[anions_df['count'] >= 10]
# print(anions_df)

# from analyses import bond_lengths
# bond_lengths.analyze_bond_length_distribution()


# df = pd.read_json('results/bonds_analysis/anions/bonds_distribution.json')
# df = df.loc[df['count'] < 30].sort_values(by='count')
# print(df)
# plt.hist(df['count'], bins=100)
# plt.show()

# from utils.bonds_length_validation import validate_bonds_lengths
# smiles = 'CCC[C-](CCN)C(C)C'
# xyz_path = Path('datasource/generated_ions/calc04_fopt/anions') / f'{smiles}.xyz'
# validate_bonds_lengths(smiles, xyz_path, 'anions')

# from utils.ions_revalidation import validate_original_ions
# validate_original_ions()

# from analyses import bond_lengths
# bond_lengths.plots.plot_bonds_charts(Path('results/bonds_analysis/anions'))
# bond_lengths.plots.plot_bonds_charts(Path('results/bonds_analysis/cations'))

# from utils import ions_revalidation
# ions_revalidation.validate_ions()



########## BOND LENGTH ANALYSIS
# from analyses import bond_lengths
# bond_lengths.run_bond_length_analysis()

########## IONS REVALIDATION
# from utils import ions_revalidation
# ions_revalidation.validate_ions()

########## RECONSTRUCTION VS PREDICTION
# from analyses import reconstruction_vs_prediction
# reconstruction_vs_prediction.analyze_reconstruction_prediction_correlation()

########## OPENBABEL OPTIMIZATION
# from analyses import openbabel_optimization
# openbabel_optimization.analyze_openbabel_optimization()

########## IONS REVALIDATION WITH HESSIAN MATRIX
# from analyses import revalidation_with_hessian_matrix
# revalidation_with_hessian_matrix.validate_ions_with_hessian_matrix()

########## OPENBABEL CONVERGENCE
# from analyses import openbabel_convergence
# openbabel_convergence.analyze()

########## OPENBABEL CONVERGENCE SERIES
# from analyses import openbabel_convergence_series
# openbabel_convergence_series.analyze()

########## REVALIDATION WITH OBABEL VS SMILES BONDS
# from analyses import revalidation_with_obabel_vs_smiles_bonds
# revalidation_with_obabel_vs_smiles_bonds.validate()
# revalidation_with_obabel_vs_smiles_bonds.filter_valid_ions()

########## REGISTRY
from registry import FileRegistry, DBRegistry, CombineRegistry

# file_registry = FileRegistry(path='test_sgvae.log')
# db_registry = DBRegistry(collection_name='test_sgvae')
# registry = CombineRegistry([file_registry, db_registry])
# registry.log({'pass': True})

import pandas as pd

df = pd.DataFrame()

df = df.append({'id':1, 'test': True}, ignore_index=True)
print(df)