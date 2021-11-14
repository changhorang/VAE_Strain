from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, args, df, n_past, n_future):
        # sliding windows
        self.X = []     # n_past 만큼의 feature 데이터
        self.y = []     # n_future 만큼의 label 데이터
                
        for i in range(args.n_past, len(df)):
            self.X.append(df[i-args.n_past:i, 1:-1])
            self.y.append(df[i-1:i, -1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]