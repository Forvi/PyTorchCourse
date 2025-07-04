import torch 
import torch.nn as nn


class LinearRegressionTorch(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        
    def forward(self, X: torch.Tensor):
        return self.linear(X)
    

class LogisticRegressionTorch(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        
    def forward(self, X: torch.Tensor):
        return self.linear(X)
    
    
class MulticlassModel(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.linear(x) 