import os
import math
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import CustomDataset
from model import Transformer_model, Transformer_encoder, GRU_model
from epoch import train_epoch, evaluate
from utils import vae_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Initializing Device: {device}')


def main(args):
    # Data Setting
    # file_path = os.path.join('C:/Users/ChangHo Kim/Documents/GitHub/VAE_Strain/data')
    # data_list = os.listdir(file_path)
    # for i in data_list:

    file_path = os.path.join(os.getcwd(), args.dataset_path)
    total_data = pd.read_csv(file_path, sep='\t')

    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    minmax_scaler = minmax_scaler.fit(total_data)
    total_data_scaled = minmax_scaler.transform(total_data)

    train_valid_split = int(len(total_data_scaled)*0.3)
    df_train = total_data_scaled[:-train_valid_split]
    df_valid = total_data_scaled[-train_valid_split:]

    train_data = CustomDataset(args, df_train, args.n_past, args.n_future)  
    valid_data = CustomDataset(args, df_valid, args.n_past, args.n_future)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers)
    
    if args.model_state == 'Transformer':
        model = Transformer_model(args, batch_size=args.batch_size, dim_embed=args.dim_embed, nhead=args.nhead, n_feature=args.n_feature, 
                                n_future=args.n_future, num_layers=args.num_layers, dropout=args.dropout).to(device)
    
    elif args.model_state == 'Transformer_encoder':
        model = Transformer_encoder(args, dim_embed=args.dim_embed, n_feature=args.n_feature, 
                                n_past=args.n_past, n_future=args.n_future,
                                num_layers=args.num_layers, dropout=args.dropout).to(device)
    
    elif args.model_state == 'GRU_model':
        model = GRU_model(args, n_feature=args.n_feature, n_past=args.n_past, n_future=args.n_future,
                        num_layers=args.num_layers, dim_model=512, dim_embed=args.dim_embed, dropout=args.dropout).to(device)

    # criterion = vae_loss()
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


    print("Start Training..")
    for epoch in range(1, args.epochs+1):
        print(f"Epoch : {epoch}/{args.epochs}")
        epoch_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {epoch_loss:.5f}")
    

    valid_loss, y_list, output_list = evaluate(model, valid_loader, criterion, device)
    rmse = np.sqrt(valid_loss)
    print(f"Validation Loss: {valid_loss:.5f}")
    print(f'RMSE is {rmse:.5f}')

    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.plot(y_list, label='target')
    plt.plot(output_list, label='prediction')

    plt.title('prediction vs target')
    plt.legend()
    
    data_path = os.path.join(os.getcwd(), "figure_save")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    plt.savefig(f"{data_path}/{args.model_state}_figure_epoch{int(args.epochs)}_past{args.n_past}_batch{args.batch_size}.png")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_state', type=str,
                        default='Transformer',
                        help='model change')

    parser.add_argument('--num_workers', default=2, type=int, 
                        help='dataloader num_workers for train')

    parser.add_argument('--dim_embed', default=512, type=int, 
                        help='transformer encoder embedding size for train')
    
    parser.add_argument('--nhead', default=8, type=int, 
                        help='transformer nhead')
    
    parser.add_argument('--num_layers', default=2, type=int, 
                        help='transformer encoder num_layers size for train')

    parser.add_argument('--dropout', default=0.1, type=float,
                        help='transformer encoder dropout ratio for train')

    parser.add_argument('--epochs', default=100, type=int, 
                        help='epochs for train')

    parser.add_argument('--lr', default=1e-5, type=float, 
                        help='optimizer learning rate for train')

    parser.add_argument('--batch_size', default=200, type=int, 
                        help='batch size for train')

    parser.add_argument('--n_past', default=30, type=int, 
                        help='n_past size for train')

    parser.add_argument('--n_future', default=1, type=int, 
                        help='n_future size for train')

    parser.add_argument('--n_feature', default=5, type=int, 
                        help='n_feature size for train')

    parser.add_argument('--dataset_path', type=str,
                        default='./data/01000002.txt',
                        help='Path of dataset')

    # parser.add_argument('--latent_size', default=2, type=int, 
    #                     help='latent_size size for VAE')

    args = parser.parse_args()


    main(args)