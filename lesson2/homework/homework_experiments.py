import torch
import torch.nn as nn
import torch.optim as optim
from models.LinearModels import LinearRegressionTorch
from homework_datasets import CsvDataset
from torch.utils.data import DataLoader
from utils.util import log_epoch

if __name__ == '__main__':
    
    # Константы для инициализации датасета
    PATH = './lesson2/homework/data/insurance.csv'
    NUM = ['age', 'bmi', 'charges', 'children']
    CAT = ['region']
    BINARY = ['sex', 'smoker']
    TARGET = 'charges'
    
    # Ключевые параметры
    LEARNING_RATE = 0.05
    EPOCHS = 100
    
    # Инициализация и обработка датасета
    dataset = CsvDataset(file_path=PATH, num_cols=NUM, cat_cols=CAT, binary_cols=BINARY, target=TARGET)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    dataset.decode_binary() # Декодирование бинарных признаков
    dataset.decode_categorical() # Декодирование категориальных
    dataset.normalize() # Нормализация 
    
    print(dataset.get_dataframe()) # Посмотреть на датасет, всё ли применилось
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {dataloader.batch_size}')
    print(f'LR: {LEARNING_RATE}')
    print(f'В модели используется:\t\nMSE, SDG, StandardScaler')
    
    # Дополнительная настройка и метрики
    features_size = dataset[0][0].shape[0]
    model = LinearRegressionTorch(in_features=features_size)
    criterion = nn.MSELoss() # MSE
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE) # SGD
    
    # Обучение
    for epoch in range(1, EPOCHS):
        total_loss = 0
        
        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_y = batch_y.unsqueeze(1)
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            optimizer.step()    
            total_loss += loss.item()
        
        avg_loss = total_loss / (i + 1)
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)