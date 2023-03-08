import numpy as np
from pathlib import Path
import json
import args_parser

########## GLOBAL VARIABLES
args = args_parser.get_args()
path = args.path

########## DEFAULT PARAMETERS
parameters = {
    # general
    'epochs': 100,
    'batch': 64,
    'valid_split': 0.1,
    'num_sampling': 1000,
    'plot_title': ['LUMO (eV)', 'HOMO (eV)'],
    'n_data_plot': 6000,

    # dataset
    'subset': 'anions',
    
    # vae general
    'latent_dim': 256,
    'max_length': 492,
    'anneal_kl': False,
    'n_cycle': 1,                    # num of cycles for KL annealing
        
    # neural networks
    'n_layers': 3,                   # num of layers for GRU decoder
    'hidden_layer_prop': 70,         # num of neurons of the integrated property network
    'hidden_layer_prop_pred': 64,   # num of neurons of the separated property network
    'learning_rate': 1e-3,
    'learning_rate_prop': 1e-4,
    'prop_weight': 1,                   
    'reconstruction_weight': 1,
    'kl_weight': 1,
    
    # property to be used in training
    'prop_pred': True,               # whether or not use property information for training
    'prop': ['lumo-fopt', 'homo-fopt'],        
    'normalization': 'none',        # how to normalize the property information (minmax, standard, none)

    # novel molecules
    'mol_generation': 'from_training',
    'num_novel_molecules': 500,
}

def load_params():
    """
    Load the parameters and hyperpameters which will be used in the model
    """
    if path:
        with open(Path(path) / 'params.json') as file:
            return json.load(file)

    return parameters