import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    """
    Датасет для MNIST.

    Args:
        train (bool): использовать тренировочный или тестовый набор (по умолчанию True)
        transform: преобразования для изображений

    Methods:
        __len__(): возвращает размер датасета
        __getitem__(idx): возвращает элемент с индексом idx 
    """
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=train, 
            download=True, 
            transform=transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class CIFARDataset(Dataset):
    """
    Датасет для CIFAR10.

    Args:
        train (bool): использовать тренировочный или тестовый набор (по умолчанию True)
        transform: преобразования для изображений

    Methods:
        __len__(): возвращает размер датасета
        __getitem__(idx): возвращает элемент с индексом idx 
    """
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=train, 
            download=True, 
            transform=transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def get_mnist_loaders(batch_size=64):
    """
    Создаёт DataLoader для MNIST с предобработкой.

    Args:
        batch_size (int): размер батча (по умолчанию 64)

    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = MNISTDataset(train=True, transform=transform)
    test_dataset = MNISTDataset(train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_cifar_loaders(batch_size=64):
    """
    Создаёт DataLoader для CIFAR-10 с предобработкой.

    Args:
        batch_size (int): размер батча (по умолчанию 64)

    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = CIFARDataset(train=True, transform=transform)
    test_dataset = CIFARDataset(train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader 