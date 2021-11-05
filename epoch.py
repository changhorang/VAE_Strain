import torch
from tqdm.auto import tqdm

def train_epoch(args, model, data_loader, criterion, optimizer, device):
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
    def evaluate(args, model, data_loader, criterion, device):
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
