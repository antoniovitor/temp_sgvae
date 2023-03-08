from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.bonds_length_validation.covalent_radii import covalent_radii_dict
from tqdm import tqdm

def _calc_covalent_radius(atom1, atom2):
    return covalent_radii_dict[atom1] + covalent_radii_dict[atom2]

def _plot_box_plots(df, save_path):
    step = 10 # divide in chunks
    chunks = [df.iloc[pos: pos+step] for pos in range(0, len(df.index), step)]

    plots_path = save_path / f'box_plots'
    plots_path.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks): # plot bonds per chunk
        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot()
        title = f'bonds {i*10}-{(i*10+step - 1)}'
        axis.set_title(title)
        axis.set_xlabel('Bond')
        axis.set_ylabel('Length')
        axis.boxplot(chunk['length'], labels=chunk['bond_id'])
        axis.grid()
        fig.savefig(plots_path / f'{title}.png')
        plt.close()

def _plot_violin_plot(df, save_path):
    step = 10 # divide in chunks
    chunks = [df.iloc[pos: pos+step] for pos in range(0, len(df.index), step)]

    plots_path = save_path / f'violin_plots'
    plots_path.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks): # plot bonds per chunk
        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot()

        title = f'bonds {i*10}-{(i*10+step - 1)}'
        labels=chunk['bond_id']

        x = range(len(chunk.index))
        length_lists = chunk['length']
        means = chunk['mean']
        stds = chunk['std']

        axis.set_title(title)
        axis.errorbar(x, means, yerr=stds * 3, fmt='o', color='red', capsize=5, label='mean ± 3*std')
        axis.errorbar(x, means, yerr=stds * 4, fmt='o', color='red', capsize=5, label='mean ± 3*std')
        axis.errorbar(x, means, yerr=stds * 5, fmt='o', color='red', capsize=5, label='mean ± 3*std')
        axis.errorbar(x, means, yerr=stds * 6, fmt='o', color='red', capsize=5, label='mean ± 3*std')
        axis.violinplot(length_lists, positions=x, showmeans=True)
        axis.set_xlabel('Bond')
        axis.set_ylabel('Length')
        axis.set_xticks(range(len(labels)), labels=labels)
        axis.legend()
        axis.grid()
        fig.savefig(plots_path / f'{title}.png')
        plt.close()

def _plot_histograms(df, save_path):
    plots_path = save_path / f'histograms'
    plots_path.mkdir(parents=True, exist_ok=True)

    for i, row in df.iterrows(): # plot bonds histograms
        length_list, mean, std = row[['length', 'mean', 'std']]

        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot()

        title = f'bond {row["bond_id"]}'
        axis.set_title(title)
        n, bins, patches = axis.hist(length_list)
        axis.vlines(x=mean, ymin=0, ymax=n.max() * 1.1, colors='red', label='mean')
        axis.vlines(x=[mean - 3*std, mean + 3*std], ymin = 0, ymax = n.max() * 1.1,
            colors='red', label='mean ± 3*std', linestyles='dashed'
        )
        axis.set_xlabel('Length')
        axis.set_ylabel('Fequency')
        axis.grid()
        axis.legend()
        fig.savefig(plots_path / f'{title}.png')
        plt.close()

def _plot_violin_plot_with_covalent_radius(df, save_path):
    step = 10 # divide in chunks
    chunks = [df.iloc[pos: pos+step] for pos in range(0, len(df.index), step)]

    plots_path = save_path / f'violin_plots_with_covalent_radius'
    plots_path.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks): # plot bonds per chunk
        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot()

        title = f'bonds {i*10}-{(i*10+step - 1)}'
        labels=chunk['bond_id']

        x = range(len(chunk.index))
        length_list = chunk['length']
        covalent_radiuses = [_calc_covalent_radius(*bond.split('-')) for bond in chunk['bond']]
                

        axis.set_title(title)
        axis.errorbar(x, covalent_radiuses, yerr=.1, fmt='o', color='red', capsize=5, label='boundaries by covalent radius')
        axis.violinplot(length_list, positions=x, showmeans=True)
        axis.set_xlabel('Bond')
        axis.set_ylabel('Length')
        axis.set_xticks(range(len(labels)), labels=labels)
        axis.legend()
        axis.grid()
        fig.savefig(plots_path / f'{title}.png')
        plt.close()


analysis_base_path = Path('results/bonds_analysis')
def plot_bonds_charts():
    analyses = [
        'bonds_from_smiles',
        'bonds_from_openbabel_GetLength',
        'bonds_from_openbabel_GetEquibLength',
    ]
    subsets = [
        'anions',
        'cations',
    ]

    for subset in tqdm(subsets, desc='subsets'):
        for analysis_name in tqdm(analyses, desc='analyses', leave=False):
            analysis_path = analysis_base_path / subset / analysis_name
            bonds_df = pd.read_json(analysis_path / 'mols_bonds.json')
            bonds_df = (
                bonds_df
                .groupby(['bond_id', 'bond', 'order'])['length']
                .agg(['mean', 'std', 'count', 'min', 'max', ('length', list)])
                .reset_index()
                .fillna(0)
                .sort_values(by='mean')
            )

            # _plot_box_plots(bonds_df, analysis_path)
            _plot_violin_plot(bonds_df, analysis_path)
            # _plot_histograms(bonds_df, analysis_path)
            # _plot_violin_plot_with_covalent_radius(bonds_df, analysis_path)


