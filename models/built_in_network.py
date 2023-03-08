import torch
import parameters
import torch.nn as nn

params = parameters.load_params()

class FeedForward(nn.Module):
    
    def __init__(self, latent_dim, hidden_size):
        super(FeedForward, self).__init__()
        
        self.feedforward1 = nn.Sequential(
             nn.Linear(latent_dim, hidden_size),
             nn.LeakyReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.LeakyReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.LeakyReLU(),
             nn.BatchNorm1d(hidden_size),
             nn.Linear(hidden_size, 1)
             )
             
        self.feedforward2 = nn.Sequential(
             nn.Linear(latent_dim, hidden_size),
             nn.LeakyReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.LeakyReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.LeakyReLU(),
             nn.BatchNorm1d(hidden_size),
             nn.Linear(hidden_size, 1)
             )

    def forward(self, x):
        output1 = self.feedforward1(x)
        output2 = self.feedforward2(x)

        return output1, output2