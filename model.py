import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

class Transformer_encoder_VAE(nn.Module):
    def __init__(self, args, dim_embed, n_feature, n_past, latent_size, n_future, num_layers, dropout):
        super(Transformer_encoder_VAE, self).__init__()

        self.n_feature = n_feature
        self.n_past = n_past
        self.latent_size = latent_size
        self.n_future = n_future
        self.num_layers = num_layers
        self.dim_embed = dim_embed
        self.dropout = dropout
        self.args = args

        self.embedding = nn.Linear(n_feature, dim_embed)
        self.pos_encoder = PositionalEncoding(dim_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embed, nhead=2, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.linear_mean1 = nn.Linear(dim_embed, latent_size*2)
        self.linear_mean2 = nn.Linear(latent_size*2, latent_size)
        self.linear_log_var1 = nn.Linear(dim_embed, latent_size*2)
        self.linear_log_var2 = nn.Linear(latent_size*2, latent_size)

        self.linear_mean = nn.Sequential(self.linear_mean1, nn.ReLU(), self.linear_mean2)
        self.linear_log_var = nn.Sequential(self.linear_log_var1, nn.ReLU(), self.linear_log_var2)

        self.decoder = nn.Linear(latent_size, n_future)
        self.decoder2 = nn.Linear(n_past, n_future)

    def encode(self, x):
        mask = self.generate_square_subsequent_mask(len(x)).to(device)
        src = self.embedding(x)*math.sqrt(self.dim_embed)
        src = self.pos_encoder(src)

        out = self.encoder_layer(src)
        out = self.transformer_encoder(out, mask)

        mean = self.linear_mean(out)
        log_var = self.linear_log_var(out)

        return mean, log_var

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var) # z : [batch_size, n_past, latent_size]

        out = self.decoder(z).transpose(2, 1)
        out = self.decoder2(out)

        # log_prob = F.log_softmax(out)

        return out, mean, log_var #, z


    def reparameterize(self, mean, log_var):
        eps = torch.randn(mean.size()).to(device)
        std = torch.exp(log_var*0.5)
        z = mean + eps*std # z : [batch_size, n_past, latent_size]

        return z

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# class VAE(nn.Module):
#     def __init__(self, encoder, decoder, n_steps=None):
#         super(VAE, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#         self.register_buffer('steps_seen', torch.tensor(0, dtype=torch.long))
#         self.register_buffer('kld_max', torch.tensor(1.0, dtype=torch.float))
#         self.register_buffer('kld_weight', torch.tensor(0.0, dtype=torch.float))
#         if n_steps is not None:
#             self.register_buffer('kld_inc', torch.tensor((self.kld_max - self.kld_weight) / (n_steps//2), dtype=torch.float))
#         else:
#             self.register_buffer('kld_inc', torch.tensor(0, dtype=torch.float))