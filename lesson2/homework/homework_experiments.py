from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.LinearModels import LinearRegressionTorch
from homework_datasets import CsvDataset
from torch.utils.data import DataLoader, random_split

if __name__ == '__main__':
    
    # Константы для инициализации датасета
    PATH = './lesson2/homework/data/insurance.csv'
    NUM = ['age', 'bmi', 'charges', 'children']
    CAT = ['region']
    BINARY = ['sex', 'smoker']
    TARGET = 'charges'
    
    LEARNING_RATE = 0.05
    EPOCHS = 100
    L1_LAMBDA = 0.001
    L2_LAMBDA = 0.001
    MAX_EPOCH = 10  # количество эпох без улучшения
    
    print(f'MSE, LBFGS, StandardScaler\nLR={LEARNING_RATE}, L1_LAMBDA={L1_LAMBDA}, L2_LAMBDA={L2_LAMBDA}\n')
    
    # Инициализация и обработка датасета
    dataset = CsvDataset(file_path=PATH, num_cols=NUM, cat_cols=CAT, binary_cols=BINARY, target=TARGET)
    dataset.decode_binary()
    dataset.decode_categorical()
    dataset.normalize()
    
    # Разделение на train и test
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Частб для early stopping
    val_size = int(0.1 * train_size)
    train_size_adjusted = train_size - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size_adjusted, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f'Размер train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')
    
    features_size = dataset[0][0].shape[0]
    model = LinearRegressionTorch(in_features=features_size)
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=1.0)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for i, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
            batch_y = batch_y.unsqueeze(1)

            def closure():
                optimizer.zero_grad()
                y_pred = model(batch_x)
                mse_loss = criterion(y_pred, batch_y)

                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())

                loss = mse_loss + L1_LAMBDA * l1_norm + L2_LAMBDA * l2_norm
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            total_loss += loss.item()

        avg_train_loss = total_loss / (i + 1)
        
        # Оценка модели
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_y = batch_y.unsqueeze(1)
                y_pred = model(batch_x)
                loss = criterion(y_pred, batch_y)
                val_loss_total += loss.item()
        avg_val_loss = val_loss_total / len(val_loader)
        
        print(f"Epoch {epoch}: train loss={avg_train_loss:.4f}, val loss={avg_val_loss:.4f}")
        
        # Проверка на улучшение
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), './lesson2/homework/models/linreg_best_earlystop2.pth')
            print(f"Loss improved, model saved!")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
        
        if epochs_no_improve >= MAX_EPOCH:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load('./lesson2/homework/models/linreg_best_earlystop2.pth'))
    print(f'\n______________________________\n')
    
    # И оценка на тестовой выборке
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_y = batch_y.unsqueeze(1)
            y_pred = model(batch_x)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            loss = criterion(y_pred, batch_y)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    print(f"Test loss: {avg_test_loss:.4f}")
    
    plt.figure(figsize=(8,6))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--', lw=2)
    plt.xlabel('True Charges')
    plt.ylabel('Predicted Charges')
    plt.title('Linear regression: LR = 0.05, L1 = 0.005, L2 = 0.005, Early stopping')
    plt.grid(True)
    plt.savefig('./lesson2/homework/plots/linreg_lbfgs_earlystopped.png') 


            