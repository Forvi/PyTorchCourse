import json
import os
import time

from utils.dataset_utils import get_cifar_loaders, get_mnist_loaders
from utils.experiment_utils import count_parameters, train_model
from utils.model_utils import FCN
from utils.visualization_utils import plot_training_history

EPOCHS = 10
LR = 0.001
BATCH_SIZE = 128
# INPUT_SIZE = 3072
INPUT_SIZE = 784
NUM_CLASSES = 10

config_path = './lesson3/homework/configs/task2/config_3layer_verybig.json'
model = FCN(config_path=config_path, input_size=INPUT_SIZE, num_classes=NUM_CLASSES)

print(model)

train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE)
# train_loader, test_loader = get_cifar_loaders(batch_size=BATCH_SIZE)

start_time = time.time()
history = train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LR, device='cpu')
end_time = time.time()

training_time = end_time - start_time
print(f"Количество параметров модели: {count_parameters(model)}")
print(f"Время обучения ({EPOCHS} эпох): {training_time:.2f} секунд")

plot_training_history(history, './lesson3/homework/plots/task2/verybig_mnist.png')


print(f"Точность на train: {history['train_accs'][-1]:.4f}")
print(f"Точность на test: {history['test_accs'][-1]:.4f}")
print(f"Loss на train: {history['train_losses'][-1]:.4f}")
print(f"Loss на test: {history['test_losses'][-1]:.4f}")