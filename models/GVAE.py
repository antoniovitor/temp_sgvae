import torch
import parameters
import torch.nn as nn
from models import grammar
from models.encoder import Encoder
from models.decoder import Decoder

device = "cuda" if torch.cuda.is_available() else "cpu"
params = parameters.load_params()

class GrammarVAE(nn.Module):
  """
  The grammarVAE model. Implementation follows the original paper with the
  addition of the 'conditional' function
  """

  def __init__(self):
    super(GrammarVAE, self).__init__()
    self.batch = params['batch']
    self.max_length = params['max_length']
    self.encoder = Encoder(latent_dim=params['latent_dim'])
    self.decoder = Decoder(latent_dim=params['latent_dim'], n_layers=params['n_layers'], max_length=self.max_length)

  def sample(self, mu, sigma):
    """Reparametrization trick to sample z"""
    sigma = torch.exp(0.5 * sigma)
    epsilon = torch.randn(params['batch'], params['latent_dim']).to(device)     
    return mu + sigma * epsilon  

  def kl(self, mu, sigma):
    """KL divergence between the approximated posterior and the prior"""
    return - 0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp()) 

  def conditional(self, x_true, x_pred):
    most_likely = torch.argmax(x_true, dim=-1)
    most_likely = most_likely.view(-1) # flatten most_likely
    ix2 = torch.unsqueeze(grammar.ind_of_ind[most_likely], -1) # index ind_of_ind with res
    ix2 = ix2.type(torch.LongTensor)
    M2 = grammar.masks[list(ix2.T)]
    M3 = torch.reshape(M2, (params['batch'], params['max_length'], grammar.D))
    P2 = torch.mul(torch.exp(x_pred), M3.float()) # apply them to the exp-predictions
    P2 = torch.divide(P2, torch.sum(P2, dim=-1, keepdims=True)) # normalize predictions
    return P2

  def forward(self, x):
    mu, sigma = self.encoder(x)
    z = self.sample(mu, sigma)
    logits = self.decoder(z)
    return z, mu, sigma, logits