import os
import json
import torch
import random
import parameters
import numpy as np
from models.GVAE import GrammarVAE
from datetime import datetime
from models.built_in_network import FeedForward
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from helpers import AnnealKL
import datasets.ions as dt
from registry import FileRegistry, CSVRegistry, DBRegistry, CombineRegistry
from pathlib import Path

#from torchinfo import summary

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
today = datetime.now()

def prop_run(params):
  saving_path = 'results/' + f'{params["subset"]}_{params["prop"][0]}_{params["prop"][1]}/' + today.strftime('%d_%m_%Y') + f'_{today.hour}_{today.minute}_{today.second}'
  Path(saving_path).mkdir(parents=True, exist_ok=True)
  
  epochs = params['epochs']             
  batch = params['batch']               
  max_length = params['max_length']  # it's the max number of rules needed to create a SMILES
  latent_dim = params['latent_dim']
  n_layers = params['n_layers']  # num of layers for GRU decoder
  hidden_layer = params['hidden_layer_prop']  # num of neurons of the property model
  min_valid_loss = np.inf

  # loading the data
  dataset = dt.Dataset(params['subset'], params['prop'])

  # splitting training and validation
  train_split, validation_split = dataset.get_encoded_splits(['train', 'val'])

  trainloader = DataLoader(train_split, batch_size=batch, drop_last=True, shuffle=False, num_workers=2, pin_memory=True)
  validloader = DataLoader(validation_split, batch_size=batch, drop_last=True, shuffle=False, num_workers=2, pin_memory=True)

  # Registries
  file_registry = FileRegistry(Path(saving_path) / 'train.vae.log')
  csv_registry = CSVRegistry(Path(saving_path) / 'train.vae.csv')
  db_registry = DBRegistry(saving_path + '.train')

  train_registry = CombineRegistry([file_registry, csv_registry, db_registry])
  
  file_registry = FileRegistry(Path(saving_path) / 'val.vae.log')
  csv_registry = CSVRegistry(Path(saving_path) / 'val.vae.csv')
  db_registry = DBRegistry(saving_path + '.val')

  validation_registry = CombineRegistry([file_registry, csv_registry, db_registry])
  
  # create model
  model = GrammarVAE()
  # print(summary(model, input_size=(params['batch'], 67, 100)))

  # load property prediction model
  pp_model = FeedForward(latent_dim, hidden_layer)
  # print(summary(pp_model, input_size=(batch, latent_dim)))
  
  if torch.cuda.is_available():
      model.cuda()
      pp_model.cuda()
      
  # optimizer and loss. The same optimizer will be used for both the VAE and the built-in feedforward neural network
  optimizer = torch.optim.Adam(list(model.parameters()) + list(pp_model.parameters()), lr=params['learning_rate'], amsgrad=True)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, min_lr=1e-6, verbose=True)
  criterion = torch.nn.BCELoss()
  pp_loss = torch.nn.MSELoss()
  
  # annealing the kl weight if needed
  if params['anneal_kl']:
      anneal = AnnealKL(n_epoch=epochs, n_cycle=params['n_cycle'], ratio=params['ratio_anneal_kl'])
  
  # saving used params for each run
  with open(os.path.join(saving_path, 'params.json'), 'w') as file:
      json.dump(params, file)
      
  # weights for the loss
  kl_weight = params['kl_weight']
  recons_weight = params['reconstruction_weight']
  prop_weight = params['prop_weight']

  for epoch in range(0, epochs):

    if params['anneal_kl']:
      beta = anneal.beta(epoch)
    
    model.train() 
    pp_model.train()
  
    avg_elbo, avg_kl, avg_recons, avg_mse_1, avg_mse_2 = 0, 0, 0, 0, 0
    for x, label_1, label_2 in trainloader:
  
      optimizer.zero_grad()
      # training procedure -----------------------------------------------------
      x = x.transpose(1, 2).contiguous().to(device)  # [batch, NUM_OF_RULES, MAX_LEN]
      z, mu, sigma, logits = model(x)

      predictions_1, predictions_2 = pp_model(z)
      
      # returning x to its original dimensions
      x = x.transpose(1, 2).contiguous()  # [batch, MAX_LEN, NUM_OF_RULES]
      x_decoded_mean = model.conditional(x, logits) 
      
      # calculating the errors
      reconstruction_loss = max_length * criterion(x_decoded_mean.view(-1), x.view(-1)) 
      kl = model.kl(mu, sigma)
      
      property_loss_1 = pp_loss(predictions_1.view(-1), label_1.to(device).float()) 
      property_loss_2 = pp_loss(predictions_2.view(-1), label_2.to(device).float()) 
      
      # annealing weigth beta to the kl
      if params['anneal_kl']:
          elbo = recons_weight * reconstruction_loss + kl_weight * kl * beta + (property_loss_1 + property_loss_2) * prop_weight
          
      else:
          elbo = recons_weight * reconstruction_loss + kl_weight * kl + (property_loss_1 + property_loss_2) * prop_weight
      
      # update parameters
      elbo.backward()
      optimizer.step()
    
      # adding the error per batch
      avg_elbo += elbo.item()
      avg_kl += kl.item()
      avg_recons += reconstruction_loss.item()
      avg_mse_1 += property_loss_1.item()
      avg_mse_2 += property_loss_2.item()
  
    # saving the results
    if params['anneal_kl']:
        log_kl = (avg_kl * beta * kl_weight)/len(trainloader)
    else:
        log_kl = (avg_kl * kl_weight)/len(trainloader)
    
    train_log = {
        'elbo': avg_elbo/len(trainloader),
        'reconstruction_loss': (avg_recons * recons_weight)/len(trainloader),
        'kl': log_kl,
        f'mse_{params["prop"][0]}': (avg_mse_1 * prop_weight)/len(trainloader),
        f'mse_{params["prop"][1]}': (avg_mse_2 * prop_weight)/len(trainloader),
    }

    train_registry.register('epoch', train_log)
    
    
################################################################################
    
    # validation procedure -----------------------------------------------------
    model.eval()
    pp_model.eval()
  
    avg_elbo_val, avg_kl_val, avg_recons_val, avg_mse_val_1, avg_mse_val_2 = 0, 0, 0, 0, 0
    
    with torch.no_grad():
        for x_val, label_val_1, label_val_2 in validloader:
      
          x_val = x_val.transpose(1, 2).contiguous().to(device)  # [batch, 76, 100]
          z_val, mu_val, sigma_val, logits_val = model(x_val)

          predictions_val_1, predictions_val_2 = pp_model(z_val)
          
          # returning x to its original dimensions
          x_val = x_val.transpose(1, 2).contiguous()  # [batch, 100, 76]
          x_decoded_mean_val = model.conditional(x_val, logits_val)  
      
          # calculating the errors
          reconstruction_loss_val = max_length * criterion(x_decoded_mean_val.view(-1), x_val.view(-1))
          kl_val = model.kl(mu_val, sigma_val) 
          
          property_loss_val_1 = pp_loss(predictions_val_1.view(-1), label_val_1.to(device).float()) 
          property_loss_val_2 = pp_loss(predictions_val_2.view(-1), label_val_2.to(device).float()) 
          
          if params['anneal_kl']:
              elbo_val = recons_weight * reconstruction_loss_val + kl_weight * kl_val * beta + (property_loss_val_1 + property_loss_val_2) * prop_weight
          else:
              elbo_val = recons_weight * reconstruction_loss_val + kl_weight * kl_val + (property_loss_val_1 + property_loss_val_2) * prop_weight
  
          # adding the error per batch
          avg_elbo_val += elbo_val.item()
          avg_kl_val += kl_val.item()
          avg_recons_val += reconstruction_loss_val.item()
          avg_mse_val_1 += property_loss_val_1.item()
          avg_mse_val_2 += property_loss_val_2.item()
      
    print(f"epoch: {epoch+1}/{epochs}\nelbo: {avg_elbo/len(trainloader):>5f}  kl: {((avg_kl * beta * kl_weight)/len(trainloader) if params['anneal_kl'] else (avg_kl * kl_weight)/len(trainloader)):>5f}  reconstruction: {(avg_recons * recons_weight)/len(trainloader):>5f}  mse_{params['prop'][0]}: {(avg_mse_1 * prop_weight)/len(trainloader):>5f}  mse_{params['prop'][1]}: {(avg_mse_2 * prop_weight)/len(trainloader):>5f} ----- elbo_val: {avg_elbo_val/len(validloader):>5f}  kl_val: {((avg_kl_val * beta * kl_weight)/len(validloader) if params['anneal_kl'] else (avg_kl_val * kl_weight)/len(validloader)):>5f}  reconstruction_val: {(avg_recons_val * recons_weight)/len(validloader):>5f}  mse_val_{params['prop'][0]}: {(avg_mse_val_1 * prop_weight)/len(validloader):>5f} mse_val_{params['prop'][1]}: {(avg_mse_val_2 * prop_weight)/len(validloader):>5f}")
    
    # saving the results
    if params['anneal_kl']:
      log_val_kl = (avg_kl_val * beta * kl_weight)/len(validloader)
    else:
      log_val_kl = (avg_kl_val * kl_weight)/len(validloader)
    
    validation_log = {
        'elbo': avg_elbo_val/len(validloader),
        'reconstruction_loss': (avg_recons_val * recons_weight)/len(validloader),
        'kl': log_val_kl,
        f'mse_{params["prop"][0]}': (avg_mse_val_1 * prop_weight)/len(validloader),
        f'mse_{params["prop"][1]}': (avg_mse_val_2 * prop_weight)/len(validloader),
    }

    validation_registry.register('epoch', validation_log)
  
    if min_valid_loss > avg_elbo_val:
      print(f'Validation loss decreased {min_valid_loss/len(validloader):.5f} ---> {avg_elbo_val/len(validloader):.5f}. Saving the model!')
      min_valid_loss = avg_elbo_val
      
      # saving the encoder and decoder separately and also the whole model
      torch.save(model.state_dict(), os.path.join(saving_path, 'gvae_model.pth'))
      # encoder
      torch.save(model.encoder.state_dict(), os.path.join(saving_path, 'gvae_encoder.pth'))
      # decoder
      torch.save(model.decoder.state_dict(), os.path.join(saving_path, 'gvae_decoder.pth'))
      # property prediction
      torch.save(pp_model.state_dict(), os.path.join(saving_path, 'prop_pred_model.pth'))
  
    write_csv(log_val, os.path.join(saving_path, 'log_val.csv'))
    
    scheduler.step(avg_elbo_val)


if __name__ == "__main__":
    params = parameters.load_params()
    print(f'Parameters being used: {params}')    
    prop_run(params)