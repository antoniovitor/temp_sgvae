import os
import json
import time
import torch
import rdkit
import pickle
import argparse
import parameters
import numpy as np
import pandas as pd
import prop_pred_model
import matplotlib.pyplot as plt
import dataset as dt

from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger 
from matplotlib import ticker
from collections import Counter
from utils import LoadSmilesData
from sklearn.manifold import TSNE
from skorch import NeuralNetRegressor
from sklearn.decomposition import PCA
from grammar_model import GrammarModel
from prop_pred_model import FeedForward
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
from pathlib import Path

random.seed(42)

RDLogger.DisableLog('rdApp.*') 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')

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

params = parameters.load_params(path)

# Loading non-normalized property information. Normalization can be done later if needed
dataset = dt.Dataset(params['subset'], params['prop'])

# SMILES/Property for training and testing
train_set = dataset.get_split('train')
smiles_train = train_set.loc[:, 'smiles'].to_list()
property_train_1 = train_set.loc[:, params['prop'][0]].to_list()
property_train_2 = train_set.loc[:, params['prop'][1]].to_list()

val_set = dataset.get_split('val')
smiles_val = val_set.loc[:, 'smiles'].to_list()
property_val_1 = val_set.loc[:, params['prop'][0]].to_list()
property_val_2 = val_set.loc[:, params['prop'][1]].to_list()

test_set = dataset.get_split('test')
smiles_test = test_set.loc[:, 'smiles'].to_list()
property_test_1 = test_set.loc[:, params['prop'][0]].to_list()
property_test_2 = test_set.loc[:, params['prop'][1]].to_list()

# property_train_1, property_train_2 = data.property_train()

model = GrammarModel(params)

if torch.cuda.is_available():
    model._encoder.cuda()
    model._decoder.cuda()
    
model._encoder.load_state_dict(torch.load(encoder_weights, map_location = device))
model._decoder.load_state_dict(torch.load(decoder_weights, map_location = device))

# If encoded data does not exists on args.path, encode SMILES for training. I'm using standard names for all the callable files
encoded_data_train = os.path.join(evaluation_path, 'encoded_data.train.pkl')
encoded_data_val = os.path.join(evaluation_path, 'encoded_data.val.pkl')

model._encoder.eval()
model._decoder.eval()

def encode_data(path, data):
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

z_train = encode_data(encoded_data_train, smiles_train)
z_val = encode_data(encoded_data_val, smiles_val)

def property_model_training():
    """
    Trains and validate the property prediction model
    INPUTS:
    z: latent space vectors
    params: file with the parameters to be used (parameters.py) 
    property_train: property values of molecules in the training set
    evaluation_path: path to save the results
    """
    
    def reset_weights(m):
        """
        Reset the model's weights for each trial training
        """
        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
            
            
    # Encoding the testing data 
    print('\nEncoding the testing data...')
    list_z = []
    with torch.no_grad():
        for smile in smiles_test:
            list_z.append(model.encode([smile]).cpu().detach().numpy())
    
    nsamples, batch, ldim = np.shape(list_z)
    list_z = np.asarray(list_z).reshape(nsamples * batch, ldim)
    list_z = torch.Tensor(list_z).to(device)
    
    errors = {}

    mae1 = []
    mse1 = []
    rmse1 = []
    
    mae2 = []
    mse2 = []
    rmse2 = []

    train_data = []
    for i in range(len(z_train)):
        train_data.append([z_train[i], property_train_1[i], property_train_2[i]])
    
    val_data = []
    for i in range(len(z_val)):
        val_data.append([z_val[i], property_val_1[i], property_val_2[i]])
        
  
    trainloader = DataLoader(train_data, batch_size=params['batch'], drop_last=True, shuffle=False)
    validloader = DataLoader(val_data, batch_size=params['batch'], drop_last=True, shuffle=False)
    
    # loading the model
    pp_model = FeedForward(params['latent_dim'], params['hidden_layer_prop_pred'])
    
    if torch.cuda.is_available():
        pp_model.cuda()
    
    optimizer = torch.optim.Adam(pp_model.parameters(), lr=params['learning_rate_prop'], amsgrad=True)
    criterion = torch.nn.MSELoss()
    
    epochs = params['epochs']
    min_valid_loss = np.inf
    
    # will be running the training and testing procedure for 5 times
    for i in range(5):
        # to reset the weights for each trial training
        pp_model.apply(reset_weights)
        valid_loss = []
        
        print(f'\n\033[1m---------- Trial training: {i+1} ----------\033[0m')
        for epoch in range(epochs):

            pp_model.train()
            avg_loss_1, avg_loss_2, avg_loss = 0, 0, 0
            for x, label_1, label_2 in trainloader:
                
                predictions_1, predictions_2 = pp_model(x.to(device))
                loss_1 = criterion(predictions_1.view(-1).to(device), label_1.float().to(device))
                loss_2 = criterion(predictions_2.view(-1).to(device), label_2.float().to(device))
                loss = loss_1 + loss_2 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                avg_loss += loss.item()
                avg_loss_1 += loss_1.item()
                avg_loss_2 += loss_2.item()

            # validation procedure -----------------------------------------------------
            pp_model.eval()
            
            avg_loss_val = 0
            
            with torch.no_grad():
                for x_val, label_val_1, label_val_2 in validloader:
                    predictions_val_1, predictions_val_2 = pp_model(x_val.to(device))
                    loss_val = criterion(predictions_val_1.view(-1).to(device), label_val_1.float().to(device)) + criterion(predictions_val_2.view(-1).to(device), label_val_2.float().to(device))
                    
                    avg_loss_val += loss_val.item()

                valid_loss.append(avg_loss_val/len(validloader))

                results = {
                    'mse': f'{avg_loss/len(trainloader):>5f}',
                    f'mse_{params["prop"][0]}': f'{avg_loss_1/len(trainloader):>5f}',
                    f'mse_{params["prop"][1]}': f'{avg_loss_2/len(trainloader):>5f}',
                    'mse_val': f'{avg_loss_val/len(validloader):>5f}")',
                }

                print(f'epoch: {epoch+1}/{epochs}')
                print(results)

        if np.mean(valid_loss) < min_valid_loss:
            
            print(f'Validation loss decreased from {min_valid_loss:.6f} to {np.mean(valid_loss):>6f}. Saving the model!')
            torch.save(pp_model.state_dict(), os.path.join(evaluation_path, 'prop_pred_model.pth'))
            
            min_valid_loss = np.mean(valid_loss)
            
        else:
            print(f'\nThe loss didn\'t decrease!')

        # after each training and validation, the model will be tested on the hold-out set
        prop_pred_normalized_1, prop_pred_normalized_2 = pp_model(list_z)

        prop_pred_1 = prop_pred_normalized_1.cpu().detach().numpy()
        prop_pred_2 = prop_pred_normalized_2.cpu().detach().numpy()
        
        mae_loss_1 = mean_absolute_error(property_test_1, prop_pred_1)
        mse_loss_1 = mean_squared_error(property_test_1, prop_pred_1)
        rmse_loss_1 = mean_squared_error(property_test_1, prop_pred_1, squared=False)
        
        mae_loss_2 = mean_absolute_error(property_test_2, prop_pred_2)
        mse_loss_2 = mean_squared_error(property_test_2, prop_pred_2)
        rmse_loss_2 = mean_squared_error(property_test_2, prop_pred_2, squared=False)
        
        mae1.append(mae_loss_1)
        mse1.append(mse_loss_1)
        rmse1.append(rmse_loss_1)
        
        mae2.append(mae_loss_2)
        mse2.append(mse_loss_2)
        rmse2.append(rmse_loss_2)

    errors[f'mae_{params["prop"][0]}'] = np.mean(mae1)
    errors[f'std_mae_{params["prop"][0]}'] = np.std(mae1)
    errors[f'mse_{params["prop"][0]}'] = np.mean(mse1)
    errors[f'std_mse_{params["prop"][0]}'] = np.std(mse1)
    errors[f'rmse_{params["prop"][0]}'] = np.mean(rmse1)
    errors[f'std_rmse_{params["prop"][0]}'] = np.std(rmse1)
    
    print(f'\n{params["prop"][0]}')
    print(f'\nThe MAE of the hold-out set is: {np.mean(mae1):.5f} \u00B1 {np.std(mae1):.5f}')
    print(f'The MSE of the hold-out set is: {np.mean(mse1):.5f} \u00B1 {np.std(mse1):.5f}')
    print(f'The RMSE of the hold-out set is: {np.mean(rmse1):.5f} \u00B1 {np.std(rmse1):.5f}')
    
    errors[f'mae_{params["prop"][1]}'] = np.mean(mae2)
    errors[f'std_mae_{params["prop"][1]}'] = np.std(mae2)
    errors[f'mse_{params["prop"][1]}'] = np.mean(mse2)
    errors[f'std_mse_{params["prop"][1]}'] = np.std(mse2)
    errors[f'rmse_{params["prop"][1]}'] = np.mean(rmse2)
    errors[f'std_rmse_{params["prop"][1]}'] = np.std(rmse2)
    
    print(f'\n{params["prop"][1]}')
    print(f'\nThe MAE of the hold-out set is: {np.mean(mae2):.5f} \u00B1 {np.std(mae2):.5f}')
    print(f'The MSE of the hold-out set is: {np.mean(mse2):.5f} \u00B1 {np.std(mse2):.5f}')
    print(f'The RMSE of the hold-out set is: {np.mean(rmse2):.5f} \u00B1 {np.std(rmse2):.5f}')
    
    with open(os.path.join(evaluation_path, 'prop_metrics.json'), 'w') as file:
            json.dump(errors, file)


if __name__ == "__main__":
    property_model_training()
