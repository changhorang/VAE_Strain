import torch
from tqdm.auto import tqdm

step = 0

def train_epoch(args, model, data_loader, criterion, optimizer, device):
    global step

    model.train()
    criterion.train()

    train_loss = 0.0
    total = len(data_loader)

    for _, (X, y) in enumerate(tqdm(data_loader)):
        X = X.float().to(device)
        y = y.float().to(device) # y : [batch_size, n_fuutre, 1]
        y = y.unsqueeze(1)

        output, mean, log_var = model(X)
        loss = criterion(output, y, mean, log_var, step)/args.batch_size
        train_loss += loss
        
        # loss = criterion(log_prob, y, mean, log_var, step)
        # loss_value = loss.item()
        # train_loss += loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

    return train_loss/total


def evaluate(args, model, data_loader, criterion, device):
    global step

    y_list = []
    output_list = []

    model.eval()
    criterion.eval()
        
    valid_loss = 0.0
    total = len(data_loader)
    with torch.no_grad():
        for _, (X, y) in enumerate(tqdm(data_loader)):
            X = X.float().to(device)
            y = y.float().to(device)
            y = y.unsqueeze(1)

            output, mean, log_var = model(X)
            loss = criterion(output, y, mean, log_var, step)/args.batch_size
            valid_loss += loss

            # loss = criterion(log_prob, y, mean, log_var, step)
            # loss_value = loss.item()
            # valid_loss += loss_value

            y_list += y.detach().reshape(-1).tolist()
            output_list += output.detach().reshape(-1).tolist()

    return valid_loss/total, y_list, output_list
