from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.util import *
from models.LinearModels import *
from homework_datasets import CsvDataset
from torch.utils.data import DataLoader, random_split


if __name__ == '__main__':
    
    # Константы для инициализации датасета
    PATH = './lesson2/homework/data/superhero_abilities_dataset.csv'
    NUM = ['Strength', 'Speed', 'Intelligence', 'Combat skill', 'Power Score', 'Popularity Score']
    CAT = ['Name', 'Universe', 'Weapon']
    BINARY = ['Alignment']
    TARGET = 'Alignment'
    
    # Ключевые параметры
    LEARNING_RATE = 0.01
    EPOCHS = 100
    
    # Инициализация и обработка датасета
    dataset = CsvDataset(file_path=PATH, num_cols=NUM, binary_cols=BINARY, cat_cols=CAT, target=TARGET)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    dataset.remove_nan() # Очистка от nan
    dataset.decode_categorical() # Декодирование категориальных признаков
    dataset.decode_binary() # Декодирование бинарных признаков
    dataset.normalize(target=TARGET)
    
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {dataloader.batch_size}')
    print(f'LR: {LEARNING_RATE}')
    print(f'В модели используется:\tSDG, CrossEntropyLoss')
    
    # Разеделние на обучающую и тестовую выборки
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f'Размер train: {len(train_dataset)}, Размер test: {len(test_dataset)}')
    
    # Дополнительная настройка и метрики
    features_size = dataset[0][0].shape[0]
    model = LinearRegressionTorch(in_features=features_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE) # SGD
    
    num_classes = int(10)
    model = MulticlassModel(in_features=features_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Обучение
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_acc = 0
        
        for i, (batch_X, batch_y) in enumerate(tqdm(train_loader)):
            batch_y = batch_y.long()
            
            optimizer.zero_grad()
            logits = model(batch_X) 
            loss = criterion(logits, batch_y) 
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch_y).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
        
        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, accuracy={avg_acc:.4f}")
    
    # Тестирование
    model.eval()
    test_loss = 0
    test_acc = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_y = batch_y.long()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch_y).float().mean().item()
            test_loss += loss.item()
            test_acc += acc
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)

    # метрики
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test loss: {avg_test_loss:.4f}, Test accuracy: {avg_test_acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # Визуализация
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('./lesson2/homework/plots/confusion_matrix1.png') 
    
    # Сохранение модели
    # torch.save(model.state_dict(), './lesson2/homework/models/linlog_mse_sgd.pth')
    