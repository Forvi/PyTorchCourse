import torch.nn as nn
import torch.nn.functional as F
import torch
    
class CNN(nn.Module):
    """ Простая сверточная нейронная сеть для MNIST.

    Args:
        input_channels (int): число входных каналов (по умолчанию 1)
        num_classes (int): число классов на выходе (по умолчанию 10)

    Attributes:
        conv1: сверточный слой (3x3), увеличивает число каналов до 32
        conv2: сверточный слой (3x3), увеличивает число каналов до 64
        pool: max pooling 2x2, уменьшает размерность в 2 раза
        fc1: полносвязный слой (128 нейронов)
        fc2: выходной полносвязный слой (num_classes)
        dropout: регуляризация Dropout (p=0.25)
    """
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): входной батч изображения

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class CustomCNN(nn.Module):
    """Простая сверточная нейросеть с настраиваемым размером ядра свертки.

    Args:
        input_channels (int): число входных каналов (по умолчанию 1)
        num_classes (int): число классов на выходе (по умолчанию 10)
        kernel_size (int): размер ядра свертки (по умолчанию 3)
    """
    def __init__(self, input_channels=1, num_classes=10, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
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


class CNN_1x1_3x3(nn.Module):
    """Сверточная нейросеть с комбинированным размером свертки 1x1 и 3x3.

    Args:
        input_channels (int): число входных каналов (по умолчанию 1)
        num_classes (int): число классов на выходе (по умолчанию 10)
    """
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=1, stride=1, padding=0) # первый сверточный слой 1x1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # второй сверточный слой 3x3
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 1x1 свертка + ReLU
        x = self.pool(F.relu(self.conv2(x)))  # 3x3 свертка + ReLU + pooling
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LayersCNN(nn.Module):
    """
    Универсальная CNN с настраиваемой глубиной.

    Args:
        input_channels (int): число входных каналов (по умолчанию 3 для CIFAR)
        num_classes (int): число классов на выходе (по умолчанию 10)
        num_conv_layers (int): количество сверточных слоев (2, 4, 6+)
        use_residual (bool): использовать ли residual-блоки (по умолчанию False)
        kernel_size (int): размер ядра свертки (по умолчанию 3)
        output_spatial_size (int): желаемый пространственный размер после адаптивного пулинга (например, 4)
    """
    def __init__(self, input_channels=3, num_classes=10, num_conv_layers=2, use_residual=False, kernel_size=3, output_spatial_size=4):
        super().__init__()
        padding = kernel_size // 2
        self.use_residual = use_residual
        # self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        channels = [input_channels, 32, 64, 128, 256, 512, 512, 1024, 1024]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((output_spatial_size, output_spatial_size))

        if use_residual:
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size, stride=1, padding=padding)
            self.bn1 = nn.BatchNorm2d(32)
            self.res_blocks = nn.ModuleList()
            self.res_blocks.append(ResidualBlock(32, 32))
            self.res_blocks.append(ResidualBlock(32, 64, stride=2))
            self.res_blocks.append(ResidualBlock(64, 64))
            
            self.fc = nn.Linear(64 * output_spatial_size * output_spatial_size, num_classes)
        else:
            conv_layers_list = []
            in_ch = input_channels
            if num_conv_layers >= 2:
                conv_layers_list.append(nn.Conv2d(in_ch, channels[1], kernel_size, stride=1, padding=padding))
                conv_layers_list.append(nn.ReLU())
                conv_layers_list.append(nn.MaxPool2d(2, 2))
                in_ch = channels[1]

            if num_conv_layers >= 4:
                conv_layers_list.append(nn.Conv2d(in_ch, channels[2], kernel_size, stride=1, padding=padding))
                conv_layers_list.append(nn.ReLU())
                conv_layers_list.append(nn.MaxPool2d(2, 2))
                in_ch = channels[2]

            if num_conv_layers >= 6:
                conv_layers_list.append(nn.Conv2d(in_ch, channels[3], kernel_size, stride=1, padding=padding))
                conv_layers_list.append(nn.ReLU())
                conv_layers_list.append(nn.MaxPool2d(2, 2))
                in_ch = channels[3]
            
            for i in range(3, num_conv_layers): 
                if i + 1 < len(channels):
                    conv_layers_list.append(nn.Conv2d(in_ch, channels[i+1], kernel_size, stride=1, padding=padding))
                    conv_layers_list.append(nn.ReLU())
                    if i < num_conv_layers -1: 
                        conv_layers_list.append(nn.MaxPool2d(2, 2)) 
                    in_ch = channels[i+1]


            self.conv_layers = nn.Sequential(*conv_layers_list)
            
            with torch.no_grad():
                dummy_input = torch.zeros(1, input_channels, 32, 32)
                dummy_output = self.conv_layers(dummy_input)
                dummy_output = self.adaptive_pool(dummy_output)
                n_size = dummy_output.view(1, -1).size(1)

            self.fc1 = nn.Linear(n_size, 128)
            self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if self.use_residual:
            x = F.relu(self.bn1(self.conv1(x)))
            for block in self.res_blocks:
                x = block(x)
            x = self.adaptive_pool(x) 
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = self.conv_layers(x)
            x = self.adaptive_pool(x) 
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNNWithResidual(nn.Module):
    """ Сверточная сеть с использованием Residual блоками.

    Args:
        input_channels (int): число входных каналов (по умолчанию 1)
        num_classes (int): число классов на выходе (по умолчанию 10)

    Attributes:
        conv1: сверточный слой 3x3 + батч-нормализация
        res1, res2, res3: три residual блока, второй с изменением размера (stride=2)
        pool: адаптивный average pooling к размеру 4x4
        fc: полносвязный выходной слой
    """
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CIFARCNN(nn.Module):
    """ Сверточная нейросеть для CIFAR-10.

    Args:
        num_classes (int): число классов на выходе (по умолчанию 10)

    Attributes:
        conv1, conv2, conv3: три сверточных слоя 3x3 с увеличением числа каналов
        pool: max pooling 2x2 для уменьшения размерности
        fc1: полносвязный слой с 256 нейронами
        fc2: выходной полносвязный слой
        dropout: Dropout с p = 0.25 для регуляризации
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): входной батч изображения

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x 