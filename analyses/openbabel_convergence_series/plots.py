import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

analysis_path = Path('results/openbabel_convergence_series')
plots_path = analysis_path / 'plots'
plots_path.mkdir(parents=True, exist_ok=True)

def plot_charts():
    df = pd.read_json(analysis_path / 'df.json')
    groups = df.groupby('mol_id')

    for mol_id in set(df['mol_id']):
        smiles_df = groups.get_group(mol_id)
        
        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot()
        title = f'Optimization convergence - smiles {mol_id}'
        axis.set_title(title)
        axis.set_ylim(bottom=0, top=6)
        axis.grid()

        for _, row in smiles_df.iterrows():
            axis.plot(row['rmsd_list'])

        fig.savefig(plots_path / f'{mol_id}.png')
        plt.close()

