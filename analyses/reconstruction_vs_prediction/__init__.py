import args_parser
import parameters
from models.GVAE import GrammarVAE
# from models.grammar_model import GrammarModel
from models.built_in_network import FeedForward
from pathlib import Path
import torch
from utils import make_dataset_grammar
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from .plots import plot_reconstruction_vs_prediction



analysis_base_path = Path('results/reconstruction_vs_prediction')
analysis_base_path.mkdir(parents=True, exist_ok=True)

def analyze_reconstruction_prediction_correlation():
    df_path = analysis_base_path / 'results.csv'
    if(df_path.exists()):
        plot_reconstruction_vs_prediction()
        return

    ########## GLOBAL VARIABLES
    args = args_parser.get_args()
    path = Path(args.path)

    params = parameters.load_params()
    batch = params['batch']               
    latent_dim = params['latent_dim']
    hidden_layer = params['hidden_layer_prop_pred']  # num of neurons of the property model
    max_length = params['max_length']  # it's the max number of rules needed to create a SMILES

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ########## DATASET
    property_dataset = pd.read_csv('datasource/temp/anions.csv')
    smiles_dataset = pd.read_csv('datasource/temp/samples/anions/novel_molecules.csv', index_col='id')

    smiles_id = property_dataset.loc[:, 'smiles'].to_list()
    smiles = [smiles_dataset.iloc[id]['smiles'] for id in smiles_id]


    smiles_one_hot = make_dataset_grammar.smiles_list_to_one_hot(smiles)
    
    properties = (
        property_dataset
            .loc[:, [params['prop'][0][:4], params['prop'][1][:4]]]
            .astype(np.float64)
            .to_numpy()
    )

    data = [[one_hot, *properties[i]] for i, one_hot in enumerate(smiles_one_hot)]
    loader = DataLoader(data, batch_size=batch, drop_last=True, shuffle=False, num_workers=2, pin_memory=True)

    ########## MODELS
    vae_model = GrammarVAE()
    pp_model = FeedForward(latent_dim, hidden_layer)

    encoder_weights = torch.load(path / 'gvae_encoder.pth', map_location=device)
    decoder_weights = torch.load(path / 'gvae_decoder.pth', map_location=device)
    pp_model_weights = torch.load(path / 'evaluation/prop_pred_model.pth', map_location=device)

    vae_model.encoder.load_state_dict(encoder_weights)
    vae_model.decoder.load_state_dict(decoder_weights)
    pp_model.load_state_dict(pp_model_weights)

    if torch.cuda.is_available():
        vae_model.encoder.cuda()
        vae_model.decoder.cuda()
        vae_model.cuda()
        pp_model.cuda()

    ########## SETTING EVALUATION
    vae_model.encoder.eval()
    vae_model.decoder.eval()
    vae_model.eval()
    pp_model.eval()

    ########## LOSS FUNCTIONS
    criterion = torch.nn.BCELoss()
    pp_loss = torch.nn.MSELoss()

    reconstruction_loss = []
    property_loss_1 = []
    property_loss_2 = []

    with torch.no_grad():
        for x, label_1, label_2 in loader:
            x = x.type(torch.cuda.FloatTensor)
            x = x.transpose(1, 2).contiguous().to(device)  # [batch, NUM_OF_RULES, MAX_LEN]

            z, mu, sigma, logits = vae_model(x)

            predictions_1, predictions_2 = pp_model(z)
            predictions_1 = predictions_1.view(-1)
            predictions_2 = predictions_2.view(-1)
            
            # returning x to its original dimensions
            x = x.transpose(1, 2).contiguous()  # [batch, MAX_LEN, NUM_OF_RULES]
            x_decoded_mean = vae_model.conditional(x, logits)

            for x_i, x_decoded_i in zip(x, x_decoded_mean):
                reconstruction_loss.append((max_length * criterion(x_decoded_i.view(-1), x_i.view(-1))).tolist())

            for pred_1_i, label_1_i in zip(predictions_1, label_1):
                property_loss_1.append((pp_loss(pred_1_i, label_1_i.to(device).float())).tolist())

            for pred_2_i, label_2_i in zip(predictions_2, label_2):
                property_loss_2.append((pp_loss(pred_2_i, label_2_i.to(device).float())).tolist())

    df = pd.DataFrame({
        'smiles': smiles[:len(reconstruction_loss)],
        'reconstruction_loss': reconstruction_loss,
        'property_loss_1': property_loss_1,
        'property_loss_2': property_loss_2,
    })

    df.to_csv(df_path)



