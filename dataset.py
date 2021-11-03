from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, args, df, n_past, n_future):
        self.X = []     # n_past 만큼의 feature 데이터
        self.y = []     # n_future 만큼의 label 데이터
        x_col = (df.shape[1]) - 1   # df 에서 -1번째 columns 까지 x
        
        for i in range(args.n_past, len(df)):
            self.X.append(df[i - args.n_past:i, 4:x_col])
            self.y.append(df[i + args.n_future - 1: i + args.n_future, x_col])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]