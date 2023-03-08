import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

analysis_path = Path('results/openbabel_optimization')

def _plot_dft_vs_openbabel_violin_charts(df):
    data = df['rmsd'].to_list()
    labels = df['smiles'].to_list()
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot()


    x = range(len(labels))
    title = 'DFT vs OpenBabel optimization'

    axis.set_title(title)
    axis.violinplot(data, positions=x, showmeans=True)
    axis.set_xlabel('SMILES')
    axis.set_ylabel('RMSD')
    axis.set_xticks(range(len(labels)), labels=x)
    axis.grid()
    fig.savefig(analysis_path / 'dft_vs_obabel.png')
    plt.close()


def _plot_openbabel_vs_openbabel_violin_charts(df):
    data = df['rmsd'].to_list()
    labels = df['smiles'].to_list()
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot()


    x = range(len(labels))
    title = 'OpenBabel vs OpenBabel optimization'

    axis.set_title(title)
    axis.violinplot(data, positions=x, showmeans=True)
    axis.set_xlabel('SMILES')
    axis.set_ylabel('RMSD')
    axis.set_xticks(range(len(labels)), labels=x)
    axis.grid()
    fig.savefig(analysis_path / 'obabel_vs_obabel.png')
    plt.close()

def plot_charts():
    dft_diff_df = pd.read_json(analysis_path / 'dft_diff.json')
    _plot_dft_vs_openbabel_violin_charts(dft_diff_df)
    
    obabel_diff_df = pd.read_json(analysis_path / 'obabel_diff.json')
    _plot_openbabel_vs_openbabel_violin_charts(obabel_diff_df)
