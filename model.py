import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

class Transformer_encoder_VAE(nn.Module):
    def __init__(self, args, dim_embed, n_feature, n_past, n_future, num_layers, dropout):
        super(Transformer_encoder_VAE, self).__init__()

        self.n_feature = n_feature
        self.n_past = n_past
        # self.latent_size = latent_size
        self.n_future = n_future
        self.num_layers = num_layers
        self.dim_embed = dim_embed
        self.dropout = dropout
        self.args = args

        self.embedding = nn.Linear(n_feature, dim_embed)
        self.pos_encoder = PositionalEncoding(dim_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embed, nhead=2, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # # mean
        # self.linear_mean1 = nn.Linear(dim_embed, latent_size*2)
        # self.linear_mean2 = nn.Linear(latent_size*2, latent_size)
        
        # # log_var
        # self.linear_log_var1 = nn.Linear(dim_embed, latent_size*2)
        # self.linear_log_var2 = nn.Linear(latent_size*2, latent_size)

        # self.linear_mean = nn.Sequential(self.linear_mean1, nn.GELU(), self.linear_mean2)
        # self.linear_log_var = nn.Sequential(self.linear_log_var1, nn.GELU(), self.linear_log_var2)

        self.decoder1 = nn.Linear(dim_embed, n_future)
        self.decoder2 = nn.Linear(n_past, n_future)

    def forward(self, x):
        mask = self.generate_square_subsequent_mask(len(x)).to(device)
        src = self.embedding(x)*math.sqrt(self.dim_embed)
        src = self.pos_encoder(src)

        out = self.encoder_layer(src)
        out = self.transformer_encoder(out, mask)

        out = self.decoder1(out).transpose(2, 1)
        out = self.decoder2(out)
        # mean = self.linear_mean(out)
        # log_var = self.linear_log_var(out)
        # mean, log_var : [batch_size, n_past, latent_size]

        # z = self.reparameterize(mean, log_var) # z : [batch_size, n_past, latent_size]

        # out = self.decoder1(z).transpose(1, 2)
        # out = self.decoder2(out) 
        # out: [batch_size, n_future]

        # log_prob = F.log_softmax(out, dim=-1).squeeze() 
        # log_prob: [batch_size, n_future, n_future]

        return out#, z, mean, log_var#, out, log_prob


    # def reparameterize(self, mean, log_var):
    #     std = torch.exp(log_var*0.5)
    #     eps = torch.randn(mean.size()).to(device)
    #     z = mean + eps*std # z : [batch_size, n_past, latent_size]

    #     return z

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Transformer_model(nn.Module):
    def __init__(self, args, batch_size, dim_embed, nhead, n_feature, n_future, num_layers, dropout):
        super(Transformer_model, self).__init__()

        self.batch_size = batch_size
        self.n_feature = n_feature
        self.n_future = n_future
        self.num_layers = num_layers
        self.dim_embed = dim_embed
        self.nhead = nhead

        self.dropout = dropout
        self.args = args
        
        self.embedding1 = nn.Linear(n_feature, dim_embed)
        self.embedding2 = nn.Linear(n_future, dim_embed)

        self.transformer = nn.Transformer(batch_first=True)
        self.decoder = nn.Linear(dim_embed, n_future)

    def forward(self, X, y):
        src_embed = self.embedding1(X)
        tgt_embed = self.embedding2(y)
        print(src_embed.shape, tgt_embed.shape)
        output = self.transformer(src_embed, tgt_embed)
        # [batch_size, n_future, dim_embed]

        output = self.decoder(output)
        # [batch_size, n_future, n_future]

        return output


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

# class generator_model(nn.Module):
#     def __init__(self, args, dim_embed, n_feature, n_past, latent_size, n_future, num_layers, dropout):
#         super(generator_model, self).__init__()

#         self.n_feature = n_feature
#         self.n_past = n_past
#         self.latent_size = latent_size
#         self.n_future = n_future
#         self.num_layers = num_layers
#         self.dim_embed = dim_embed
#         self.dropout = dropout
#         self.args = args
        
#         self.embedding = nn.Linear(latent_size, dim_embed)
#         self.pos_encoder = PositionalEncoding(dim_embed)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embed, nhead=2, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

#         self.decoder_1 = nn.Linear(dim_embed, dim_embed//2)
#         self.decoder_2 = nn.Linear(dim_embed//2, n_future)

#         self.decoder1 = nn.Sequential(self.decoder_1, nn.ReLU(), self.decoder_2)
#         self.decoder2 = nn.Linear(n_past, n_future)

#     def forward(self, z):
#         mask = self.generate_square_subsequent_mask(len(z)).to(device)
#         src = self.embedding(z)*math.sqrt(self.dim_embed)
#         src = self.pos_encoder(src)

#         out = self.encoder_layer(src)
#         out = self.transformer_encoder(out, mask)

#         out = self.decoder1(out).transpose(1, 2)
#         out = self.decoder2(out) 
#         # out: [batch_size, n_future]

#         # log_prob = F.log_softmax(out, dim=-1).squeeze() 
#         # log_prob: [batch_size, n_future, n_future]

#         return out

#     def generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

# class vae_model(nn.Module):
#     def __init__(self, args, dim_embed, n_feature, n_past, latent_size, n_future, num_layers, dropout):
#         super(vae_model, self).__init__()
#         self.encode = Transformer_encoder_VAE(args, dim_embed, n_feature, n_past, latent_size, n_future, num_layers, dropout)
#         self.decode = generator_model(args, dim_embed, n_feature, n_past, latent_size, n_future, num_layers, dropout)

#     def forward(self, x):
#         z, mean, log_var = self.encode(x)
#         # z_ = torch.randn(z.size(0), z.size(1), z.size(3)).to(device)
#         out = self.decode(z)

#         return out, mean, log_var

# class discriminator_model(nn.Module):
#     def __init__(self):
#         super(self, discriminator_model).__init__()

