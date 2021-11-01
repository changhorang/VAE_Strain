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

from model import Transformer_encoder

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    criterion.train()

    train_loss = 0.0
    total = len(data_loader)

    for _, (X, y) in enumerate(tqdm(data_loader)):
        X = X.float().to(device)
        y = y.float().to(device)
        y = y.unsqueeze(1)

        output = model(X)
        loss = criterion(output, y)
        loss_value = loss.item()
        train_loss += loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

    return train_loss/total

with torch.no_grad():
    def evaluate(model, data_loader, criterion, device):
        y_list = []
        output_list = []

        model.eval()
        criterion.eval()
        
        valid_loss = 0.0
        total = len(data_loader)

        for _, (X, y) in enumerate(tqdm(data_loader)):
            X = X.float().to(device)
            y = y.float().to(device)
            y = y.unsqueeze(1)

            output = model(X)
            loss = criterion(output, y)
            loss_value = loss.item()
            valid_loss += loss_value

            y_list += y.detach().reshape(-1).tolist()
            output_list += output.detach().reshape(-1).tolist()

        return valid_loss/total, y_list, output_list

def main():
    # Data Setting
    # file_path = os.path.join('C:/Users/ChangHo Kim/Documents/GitHub/VAE_Strain/data')
    # data_list = os.listdir(file_path)
    # for i in data_list:

    file_path = os.path.join(os.getcwd(), './data/01000002.txt')
    total_data = pd.read_csv(file_path, sep='\t')

    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
    minmax_scaler = minmax_scaler.fit(total_data)
    total_data_scaled = minmax_scaler.transform(total_data)

    train_valid_split = int(len(total_data_scaled) * 0.3)
    df_train = total_data_scaled[:-train_valid_split]
    df_valid = total_data_scaled[-train_valid_split:]

    train_data = CustomDataset(df_train, n_past, n_future)  
    valid_data = CustomDataset(df_valid, n_past, n_future)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=2)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, num_workers=2)
    
    model = Transformer_encoder(dim_embed=256, n_feature=2, n_past=n_past, n_future=1, num_layers=6, dropout=0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


    # Train
    epochs = 50 # 30 이상에서 더이상 학습이 되지 않음
    print("Start Training..")
    for epoch in range(1, epochs):
        print(f"Epoch : {epoch}/{epochs}")
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {epoch_loss:.5f}")
    

    valid_loss, y_list, output_list = evaluate(model, valid_loader, criterion, device)
    rmse = np.sqrt(valid_loss)
    print(f"Validation Loss: {valid_loss:.5f}")
    print(f'RMSE is {rmse:.5f}')
    
    plt.clf()
    plt.figure(figsize=(10, 8))
    plt.plot(y_list)
    plt.plot(output_list)
    data_path = os.path.join(os.getcwd(), "figure_save")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    plt.savefig(f"{data_path}/figure_{int(epoch)+1}_{n_past}_{batch_size}.png")
    # print(output_list)


if __name__ == "__main__":
    main()