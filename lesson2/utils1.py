import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
        
        
    def __len__(self):
        return len(self.X)
    
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

def make_data(n=100, noise=0.1, source='random', type='regression'):
    if source == 'random' and type == 'regression':
        X = torch.randn(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes' and type == 'regression':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    elif source == 'random' and type == 'classification':
        X = torch.rand(n, 2)
        w = torch.tensor([2.0, -3.0])
        b = 0.5
        logits = X @ w + b
        y = (logits > 0).float().unsqueeze(1)
        return X, y
    elif source == 'breast_cancer' and type == 'classification':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Invalid source and type') 
    
    
def mse(y_true, y_pred):
    return ((y_pred - y_true) ** 2).mean().item()
    

def log_epoch(epoch, loss, **metrics):
    msg = f"Epoch {epoch}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)
    