import torch.nn as nn


class FCN(nn.Module):
    """
    Полносвязная нейронная сеть.

    Args:
        input_size (int): размер входного вектора
        num_classes (int): число классов на выходе
    """
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.layers = self._build_layers()
    
    def _build_layers(self):
        """ Инициализация контейнера слоев

        Returns:
            torch.nn.Sequential: контейнер слоев
        """
        layers = []
        prev_size = self.input_size

        layers.append(nn.Linear(prev_size, 512))
        layers.append(nn.BatchNorm1d(512))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.5))
        prev_size = 512

        layers.append(nn.Linear(prev_size, 256))
        layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.5))
        prev_size = 256

        layers.append(nn.Linear(prev_size, self.num_classes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): входной батч изображения

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        x = x.view(x.size(0), -1)
        return self.layers(x)
