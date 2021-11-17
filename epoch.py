import torch
from tqdm.auto import tqdm

# step = 0

def train_epoch(args, model, data_loader, criterion, optimizer, device, model_state):
    # global step

    model.train()
    criterion.train()

    train_loss = 0.0
    total = len(data_loader)

    for _, (X, y) in enumerate(tqdm(data_loader)):
        X = X.float().to(device)
        y = y.float().to(device) # y : [batch_size, n_fuutre]
        if model_state != 'GRU_model':
            y = y.unsqueeze(1)

        # output, mean, log_var = model(X)
        # loss = criterion(output, y, mean, log_var, step)
        if args.model_state == 'Transformer_model':
            output = model(X, y)
        else:
            output = model(X)
        loss = criterion(output, y)
        loss_value = loss.item()
        train_loss += loss_value
        
        # loss = criterion(log_prob, y, mean, log_var, step)
        # loss_value = loss.item()
        # train_loss += loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # step += 1

    return train_loss/total


def evaluate(args, model, data_loader, criterion, device, model_state):
    # global step

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
            if model_state != 'GRU_model':
                y = y.unsqueeze(1)

            # output, mean, log_var = model(X)
            # loss = criterion(output, y, mean, log_var, step)
            if args.model_state == 'Transformer_model':
                output = model(X, y)
            else:
                output = model(X)
            loss = criterion(output, y)
            loss_value = loss.item()
            valid_loss += loss_value

            # loss = criterion(log_prob, y, mean, log_var, step)
            # loss_value = loss.item()
            # valid_loss += loss_value

            y_list += y.detach().reshape(-1).tolist()
            output_list += output.detach().reshape(-1).tolist()

    return valid_loss/total, y_list, output_list
