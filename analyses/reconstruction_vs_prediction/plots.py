import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import parameters

analysis_base_path = Path('results/reconstruction_vs_prediction')
plots_path = analysis_base_path / 'plots'
plots_path.mkdir(parents=True, exist_ok=True)

def plot_reconstruction_vs_prediction():
    df = pd.read_csv(analysis_base_path / 'results.csv')
    params = parameters.load_params()
    
    reconstruction_loss = df['reconstruction_loss']
    prop_name_1 = params['prop'][0][:4]
    prediction_lost_1 = df['property_loss_1']
    prop_name_2 = params['prop'][1][:4]
    prediction_lost_2 = df['property_loss_2']


    # Property 1
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot()
    title = f'Reconstruction-{prop_name_1} error correlation'
    axis.set_title(title)
    axis.scatter(prediction_lost_1, reconstruction_loss)

    axis.set_xlabel(f'{prop_name_1} prediction error')
    axis.set_ylabel('Reconstruction error')
    axis.grid()
    fig.savefig(plots_path / f'{title}.png')
    plt.close()   

    # Property 2
    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot()
    title = f'Reconstruction-{prop_name_2} error correlation'
    axis.set_title(title)
    axis.scatter(prediction_lost_2, reconstruction_loss)

    axis.set_xlabel(f'{prop_name_2} prediction error')
    axis.set_ylabel('Reconstruction error')
    axis.grid()
    fig.savefig(plots_path / f'{title}.png')
    plt.close()    

