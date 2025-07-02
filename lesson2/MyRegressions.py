import torch
from lesson2.utils1 import *

class MyLinearRegression:
    def __init__(self, in_features):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=None)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=None)
        
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.__call__(X)
    
    def backward(self, X: torch.Tensor, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = X.size(0)
        self.dw = -1 / n * X.T @ (y - y_pred)
        self.db = -(y - y_pred).mean()
        
        
    def step(self, lr):
        self.w += lr * self.dw
        self.b += lr * self.db
        
    
    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)
        

if __name__ == '__main__':
    EPOCHS = 100
    X, y = make_data(n=1000, noise=0.1)
    
    dataset = MyDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    print(f'Пример данных: {dataset[0]}')
    
    model = MyLinearRegression(in_features=1)
    lr = .05
    
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            y_pred = model(batch_X)
            loss = mse(y_pred, batch_y)
            total_loss += loss
            
            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)
            model.step(lr)
        
        avg_loss = total_loss / (i + 1)
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)