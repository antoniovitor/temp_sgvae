import os
import torch
import pickle
import argparse
import parameters
import pandas as pd

from rdkit import RDLogger 
from models.grammar_model import GrammarModel
from models.prop_pred_model import FeedForward
from sklearn.metrics import mean_absolute_error
import random
from pathlib import Path

random.seed(42)

RDLogger.DisableLog('rdApp.*') 

device = "cuda" if torch.cuda.is_available() else "cpu"

# Parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path to the weights file', type=str)
parser.add_argument('--plot', help='Plot the latent space', action='store_true')
parser.add_argument('--evaluation', help='Evaluate the model on prior validity, novelty and uniqueness', action='store_true')
parser.add_argument('--train_property', help='Train and evaluate the property prediction model', action='store_true')
parser.add_argument('--test_property', help='Test the property prediction model in a hold-out set', action='store_true')
parser.add_argument('--hyper_optim', help='Perform hyperparameter optimization over the property prediction model', action='store_true')
parser.add_argument('--sample_molecules', help='Sample novel molecules according to some criteria', action='store_true')
parser.add_argument('--reconstruction', help='Estimates the reconstruction accuracy', action='store_true')
parser.add_argument('--active_units', help='Calculates the number of active units', action='store_true')
parser.add_argument('--latent_traversal', help='Decodes novel molecules based on changes in a specific latent dimension', action='store_true')
parser.add_argument('--property_estimation', help='Predicts the choosen properties of the input molecules', action='store_true')
args = parser.parse_args()
path = args.path

# Folder to save the evaluations
evaluation_path = os.path.join(path, 'evaluation')
if not os.path.exists(evaluation_path):
    os.makedirs(evaluation_path)

# loading the weights for the encoder and decoder
encoder_weights = os.path.join(path, 'gvae_encoder.pth')
decoder_weights = os.path.join(path, 'gvae_decoder.pth')

params = parameters.load_params()

subset = params['subset']

model = GrammarModel(params)

if torch.cuda.is_available():
    model._encoder.cuda()
    model._decoder.cuda()
    
model._encoder.load_state_dict(torch.load(encoder_weights, map_location = device))
model._decoder.load_state_dict(torch.load(decoder_weights, map_location = device))

model._encoder.eval()
model._decoder.eval()

def encode_data(data, path=None):
    if path == None:
        z = model.encode(data).cpu().detach().numpy()
        return z

    if not os.path.exists(path):
        # Encoding the training data
        print('\nEncoding the training data. This might take some time...\n')
        z = model.encode(data).cpu().detach().numpy()

        with open(path, 'wb') as file:
            pickle.dump(z, file)

    else:
        # Loading the encoded data
        with open(path, 'rb') as file:
            z = pickle.load(file)
            
        print('\nThe encoded data already exists and has been loaded!\n')
    
    return z



def test_evaluation():
    pp_model = FeedForward(params['latent_dim'], params['hidden_layer_prop_pred'])
    pp_model_weights = os.path.join(evaluation_path, 'prop_pred_model.pth')
    pp_model.load_state_dict(torch.load(pp_model_weights, map_location=device))

    df_prop = pd.read_csv(f'samples/{subset}/{subset}_with_prop.csv')
    df_smiles = pd.read_csv(f'samples/{subset}/novel_molecules.csv')
    property_test_1 = df_prop.loc[:, params['prop'][0]].to_list()
    property_test_2 = df_prop.loc[:, params['prop'][1]].to_list()

    ids = df_prop.loc[:, 'smiles'].to_list()
    smiles = [df_smiles.iloc[id]['smiles'] for id in ids]

    list_z = torch.Tensor(encode_data(smiles))
    pp_model.eval()
    with torch.no_grad():
        prop_pred_normalized_1, prop_pred_normalized_2 = pp_model(list_z)
    prop_pred_1 = prop_pred_normalized_1.cpu().detach().numpy()
    prop_pred_2 = prop_pred_normalized_2.cpu().detach().numpy()

    mae_loss_1 = mean_absolute_error(property_test_1, prop_pred_1)
    mae_loss_2 = mean_absolute_error(property_test_2, prop_pred_2)

    print(f'Subset: {subset}')
    print(f'\n{params["prop"][0]}')
    print(f'\nThe MAE of the hold-out set is: {mae_loss_1:.5f}')
    print(f'\n{params["prop"][1]}')
    print(f'\nThe MAE of the hold-out set is: {mae_loss_2:.5f}')



if __name__ == "__main__":
    test_evaluation()