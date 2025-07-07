import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomConv2d(nn.Module):
    """
    Кастомный сверточный слой с логикой масштабирования.

    Args:
        in_channels (int): число входных каналов.
        out_channels (int): число выходных каналов.
        kernel_size (int): размер ядра свертки.
        stride (int, optional): шаг свертки (по умолчанию 1).
        padding (int, optional): паддинг (по умолчанию 0).

    Attributes:
        conv (nn.Conv2d): стандартный сверточный слой.
        scale (nn.Parameter): коэффициент для масштабирования выхода слоя.

    Forward:
        x (torch.Tensor): входной тензор.
        return: выходной тензор с масштабированием.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = self.conv(x)
        out = out * self.scale
        return out


class SpatialAttention(nn.Module):
    """
    Пространственный attention-механизм для CNN.

    Args:
        in_channels (int): число входных каналов.

    Attributes:
        conv (nn.Conv2d): сверточный слой для генерации attention-карты.
        sigmoid (nn.Sigmoid): сигмоид для нормализации карты внимания.

    Forward:
        x (torch.Tensor): входной тензор формы (batch, in_channels, H, W).
        return: тензор после применения attention-маски.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_map = self.sigmoid(self.conv(x))
        return x * attn_map


class Swish(nn.Module):
    """
    Функция активации Swish (x * sigmoid(x)).

    Forward:
        x (torch.Tensor): входной тензор.
        return: выходной тензор после Swish-активации.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class GeMPooling(nn.Module):
    """
    Кастомный pooling слой Generalized Mean Pooling (GeM).

    Args:
        p (float, optional): начальное значение степени p (по умолчанию 3.0).
        eps (float, optional): малое число для избежания деления на ноль (по умолчанию 1e-6).

    Attributes:
        p (nn.Parameter): обучаемый параметр степени.
        eps (float): стабильность вычислений.

    Forward:
        x (torch.Tensor): входной тензор формы (batch, channels, H, W).
        return: pooled-тензор формы (batch, channels, 1, 1).
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), x.size()[-2:]).pow(1.0 / self.p)
    
    
    
# models

class StandardCNN(nn.Module):
    """Базовая CNN только со стандартными слоями."""
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 32, 32)
            dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
            dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
            n_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(n_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CustomConvCNN(nn.Module):
    """StandardCNN, но с CustomConv2d."""
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = CustomConv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = CustomConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 32, 32)
            dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
            dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
            n_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(n_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AttentionCNN(nn.Module):
    """StandardCNN, но с SpatialAttention."""
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.attn1 = SpatialAttention(32) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.attn2 = SpatialAttention(64) 
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 32, 32)
            dummy_output = self.pool(self.attn1(F.relu(self.conv1(dummy_input))))
            dummy_output = self.pool(self.attn2(F.relu(self.conv2(dummy_output))))
            n_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(n_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.attn1(x) 
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.attn2(x) 
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SwishCNN(nn.Module):
    """StandardCNN, но с Swish активацией."""
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.swish_act = Swish() # Кастомная активация
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 32, 32)
            dummy_output = self.pool(self.swish_act(self.conv1(dummy_input)))
            dummy_output = self.pool(self.swish_act(self.conv2(dummy_output)))
            n_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(n_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.swish_act(self.conv1(x))
        x = self.pool(x)
        x = self.swish_act(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GeMCNN(nn.Module):
    """StandardCNN, но с GeMPooling."""
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.gem_pool = GeMPooling()
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 32, 32)
            dummy_output = self.gem_pool(F.relu(self.conv1(dummy_input))) 
            dummy_output = self.gem_pool(F.relu(self.conv2(dummy_output)))
            n_size = dummy_output.view(1, -1).size(1) 

        self.fc1 = nn.Linear(n_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.gem_pool(x) 
        x = F.relu(self.conv2(x))
        x = self.gem_pool(x) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AllCustomCNN(nn.Module):
    """Модель, использующая все реализованные кастомные слои."""
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = CustomConv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.attn1 = SpatialAttention(32)
        self.act1 = Swish() 
        self.conv2 = CustomConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.attn2 = SpatialAttention(64)
        self.act2 = Swish() 
        self.pool = GeMPooling() 
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 32, 32)
            dummy_output = self.act1(self.attn1(self.conv1(dummy_input)))
            dummy_output = self.act2(self.attn2(self.conv2(dummy_output)))
            dummy_output = self.pool(dummy_output) 
            n_size = dummy_output.view(1, -1).size(1) 

        self.fc1 = nn.Linear(n_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.act1(self.attn1(self.conv1(x)))
        x = self.act2(self.attn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
