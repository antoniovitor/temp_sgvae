import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

analysis_path = Path('results/openbabel_convergence')
plots_path = analysis_path / 'plots'
plots_path.mkdir(parents=True, exist_ok=True)

def plot_charts():
    df = pd.read_json(analysis_path / 'dft_diff.json')

    for i, row in df.iterrows():
        data, smiles = row[['rmsd', 'smiles']]
        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot()
        title = f'Optimization convergence - smiles {i}: {smiles}'

        axis.set_title(title)
        axis.plot(data)
        axis.set_ylim(0, 6)
        axis.grid()
        fig.savefig(plots_path / f'{i}.png')
        plt.close()
