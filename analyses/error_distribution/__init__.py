import args_parser 
import parameters
from pathlib import Path
from models.grammar_model import GrammarModel
import datasets.ions as dt
import torch
import pickle
import utils.smile_validation as smile_validation
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from .plots import plot_error_distribution

########## FUNCTIONS
def analyze_error_distribution():
    ########## SEED
    random.seed(42)

    ########## GLOBAL VARIABLES
    args = args_parser.get_args()
    path = Path(args.path)
    split = args.split

    params = parameters.load_params()
    subset = params['subset']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    error_distribution_path = path / f'error_distribution/{split}/'
    if not error_distribution_path.exists(): error_distribution_path.mkdir(parents=True)

    ########## DATASET
    dataset = dt.Dataset(params['subset'], params['prop'])
    train_set, val_set, test_set = dataset.get_splits(['train', 'val', 'test'])

    smiles_train = train_set.loc[:, 'smiles'].to_list()
    smiles_val = val_set.loc[:, 'smiles'].to_list()
    smiles_test = test_set.loc[:, 'smiles'].to_list()

    existing_smiles = smiles_train + smiles_val + smiles_test
    smiles_set = dataset.get_split(split).loc[:, 'smiles'].to_list()
    random.shuffle(smiles_set)
    smiles_set = smiles_set[:500]

    ########## MODEL
    # TODO: rewrite model do encapsulate weights loading, training, encoding and decoding
    model = GrammarModel(params)

    if torch.cuda.is_available(): # move models to CUDA if available TODO: remove and include in the model class
        model._encoder.cuda()
        model._decoder.cuda()

    # load models weights
    model._encoder.load_state_dict(torch.load(path / 'gvae_encoder.pth', map_location = device))
    model._decoder.load_state_dict(torch.load(path / 'gvae_decoder.pth', map_location = device))
    # activate evaluation mode in model
    model._encoder.eval()
    model._decoder.eval()

    ########## ENCODING SMILES
    encoded_path = error_distribution_path / f'encoded_data.{split}.pkl'
    if encoded_path.exists():
        with open(encoded_path, 'rb') as file:
            z_set = pickle.load(file)
    else:
        print(f'Encoding {split} data...')
        z_set = model.encode(smiles_set).cpu().detach().numpy()
        with open(encoded_path, 'wb') as file:
            pickle.dump(z_set, file)

    z_set = torch.Tensor(np.array([[i] for i in z_set])).to(device)

    ########## LOADING DATAFRAME
    dataframe_path = error_distribution_path / 'data.csv'
    if dataframe_path.exists():
        df = pd.read_csv(dataframe_path, index_col='id')
    else:
        df = pd.DataFrame(columns=['id', 'success'])
        df.set_index('id', inplace=True)

    ########## ANALYZING ERROR DISTRIBUTION
    with torch.no_grad():
        for z in tqdm(z_set[len(df):]):
            errors = { 'success': 0 }
            for i in range(100):
                smile = model.decode(z)[0]
                try:
                    validation = smile_validation.validate_smiles(smile, subset, existing_smiles)
                except:
                    print(f'Unknown error while validating smile "{smile}"')
                    errors['unknown'] = errors.get('unknown', 0) + 1
                    continue

                if validation['result']:
                    errors['success'] += 1
                else:
                    errors[validation['error']] = errors.get(validation['error'], 0) + 1
            for column in errors.keys():
                if column not in df.columns: df[column] = 0
            df.loc[len(df.index)] = errors
            df.to_csv(dataframe_path)

    split_set = dataset.get_split(split)

    if 'smiles' not in df.columns: df.insert(0, 'smiles', smiles_set)
    # if 'homo-fopt' not in df.columns: df.insert(1, 'homo-fopt', split_set.loc['homo-fopt'])
    # if 'lumo-fopt' not in df.columns: df.insert(2, 'lumo-fopt', split_set.loc['lumo-fopt'])
    df.to_csv(dataframe_path)

    df['z'] = z_set.tolist()
    df.to_json(error_distribution_path / 'data_with_latent_space.csv')


    ########## ENCODING SMILES
    encoded_path = error_distribution_path / f'encoded_data.{split}.complete.pkl'
    if encoded_path.exists():
        with open(encoded_path, 'rb') as file:
            z_space = pickle.load(file)
    else:
        smiles_set = dataset.get_split(split).loc[:, 'smiles'].to_list()
        random.shuffle(smiles_set)
        smiles_set = smiles_set[:5000]
        print(f'Encoding complete {split} data...')
        z_space = model.encode(smiles_set).cpu().detach().numpy()
        with open(encoded_path, 'wb') as file:
            pickle.dump(z_space, file)
    plot_error_distribution(df, z_space, error_distribution_path / 'plots')

