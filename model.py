import os
from tqdm import tqdm
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Transformer_encoder(nn.Module):
    def __init__(self, args, dim_embed, n_feature, n_past, n_future, num_layers, dropout,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(Transformer_encoder, self).__init__()

        self.n_feature = n_feature
        self.n_past = n_past
        self.n_future = n_future
        self.num_layers = num_layers
        self.dim_embed = dim_embed
        self.dropout = dropout

        self.args = args
        self.device = device

        self.embedding = nn.Linear(n_feature, dim_embed)

        self.pos_encoder = PositionalEncoding(dim_embed)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embed, nhead=2, dropout=dropout)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(dim_embed, n_future)

        self.decoder2 = nn.Linear(n_past, n_future)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        mask = self.generate_square_subsequent_mask(len(x)).to(device)
        src = self.embedding(x)*math.sqrt(self.dim_embed)
        src = self.pos_encoder(src)

        out = self.encoder_layer(src)
        out = self.transformer_encoder(out, mask)
    
        out = self.decoder(out).transpose(1, 2)
        out = self.decoder2(out)

        return out


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