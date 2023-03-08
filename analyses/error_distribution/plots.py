from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import args_parser
from pathlib import Path

errors_columns = [
    'SMILE_DOES_NOT_EXIST',
    'RDKIT_CAN_NOT_READ_SMILE',
    'SMILE_ALREADY_EXISTS_IN_DATASET',
    'FORMAL_CHARGE_OF_ANION_IS_NOT_-1',
    'SPIN_IS_NOT_ZERO',
    'GEOMETRY_DID_NOT_CONVERGE',
    'HESSIAN_MATRIX_HAVE_NEGATIVES_EIGENVALUES',
]

properties_list = errors_columns + ['success', 'unknown']

def plot_error_distribution(df, z_space, path):
    ########## PCA ANALYSIS
    z_error = np.array([i[0] for i in df['z'].to_list()])
    z_plot = np.concatenate([z_error, z_space])
    
    pca = PCA(n_components=2, random_state=1)
    pca_data = pca.fit_transform(z_plot)

    pca_error = pca_data[:500]
    pca_space = pca_data[500:]

    ########## PLOTS
    plots_path = path
    plots_path.mkdir(parents=True, exist_ok=True)
    for i, prop in enumerate(properties_list):
        error_list = df[prop].to_numpy()
        color_map =  error_list

        fig = plt.figure(figsize=(15,10))
        axis = fig.add_subplot()


        axis.scatter(pca_space[:, 0], pca_space[:, 1], c='#555', label='Latent space')
        errors_plot = axis.scatter(pca_error[:, 0], pca_error[:, 1], c=color_map, cmap='viridis', label='Validation error', vmin=0, vmax=100)
        
        # Color bar settings
        color_bar = plt.colorbar(errors_plot)
        color_bar.set_ticks([0, 100])
        color_bar.set_ticklabels(['0%', '100%'])
        color_bar.set_label('Validation error')
        
        axis.set_title(prop)
        axis.set_xlabel('PCA 1')
        axis.set_ylabel('PCA 2')
        axis.legend()
        fig.savefig(plots_path / f'{i+1}. {prop}.png')
