import time
import pandas as pd
import json
from utils.visualization_utils import plot_training_history
from utils.comparison_utils import get_cifar_loaders, get_mnist_loaders
from models.fc_models import FC
from models.cnn_models import CIFARCNN, CNN, CNN_1x1_3x3, CustomCNN
from utils.training_utils import count_parameters, train_model
import logging


if __name__ == "__main__":
    EPOCHS = 10
    logging.basicConfig(level=logging.INFO, filename="./lesson4/homework/results/architecture_analysis/cifar_fc_log.log",filemode="w")
    
    try:
        logging.info('Getting the dataset')
        train_loader, test_loader = get_cifar_loaders(batch_size=128)
    except:
        logging.exception('Failed to get dataset')
    
    
    # свертка размера 3x3
    try:
        logging.basicConfig(level=logging.INFO, filename="./lesson4/homework/results/architecture_analysis/cifar_cnn_log.log",filemode="w")
        logging.debug('CNN IS STARTING...')
        
        model_3 = CustomCNN(kernel_size=3, input_channels=3)
        
        print(model_3)
        parametrs_3 = count_parameters(model_3)

        logging.info('Train model')
        start_time = time.time()
        history_3 = train_model(model_3, train_loader, test_loader, epochs=EPOCHS, lr=0.001, device='cpu')
        end_time = time.time()
        training_time_3 = end_time - start_time

        plot_training_history(history=history_3, save_path='./lesson4/homework/plots/cifar_kernel_3.png')
    except Exception as e:
        logging.error('An unexpected error occurred')
        raise
    
    
    # свертка размера 5x5
    try:
        logging.basicConfig(level=logging.INFO, filename="./lesson4/homework/results/architecture_analysis/cifar_cnn_log.log",filemode="w")
        logging.debug('CNN IS STARTING...')
        
        model_5 = CustomCNN(kernel_size=5, input_channels=3)
        
        print(model_5)
        parametrs_5 = count_parameters(model_5)

        logging.info('Train model')
        start_time = time.time()
        history_5 = train_model(model_5, train_loader, test_loader, epochs=EPOCHS, lr=0.001, device='cpu')
        end_time = time.time()
        training_time_5 = end_time - start_time

        plot_training_history(history=history_5, save_path='./lesson4/homework/plots/cifar_kernel_5.png')
    except Exception as e:
        logging.error('An unexpected error occurred')
        raise
    
    
    
    # свертка размера 7x7
    try:
        logging.basicConfig(level=logging.INFO, filename="./lesson4/homework/results/architecture_analysis/cifar_cnn_log.log",filemode="w")
        logging.debug('CNN IS STARTING...')
        
        model_7 = CustomCNN(kernel_size=7, input_channels=3)
        
        print(model_7)
        parametrs_7 = count_parameters(model_7)

        logging.info('Train model')
        start_time = time.time()
        history_7 = train_model(model_7, train_loader, test_loader, epochs=EPOCHS, lr=0.001, device='cpu')
        end_time = time.time()
        training_time_7 = end_time - start_time

        plot_training_history(history=history_7, save_path='./lesson4/homework/plots/cifar_kernel_7.png')
    except Exception as e:
        logging.error('An unexpected error occurred')
        raise
    
    
    # свертка размера 1x1 + 3x3
    try:
        logging.basicConfig(level=logging.INFO, filename="./lesson4/homework/results/architecture_analysis/cifar_cnn_log.log",filemode="w")
        logging.debug('CNN IS STARTING...')
        
        model_1_3 = CNN_1x1_3x3(input_channels=3)
        
        print(model_1_3)
        parametrs_1_3 = count_parameters(model_1_3)

        logging.info('Train model')
        start_time = time.time()
        history_1_3 = train_model(model_1_3, train_loader, test_loader, epochs=EPOCHS, lr=0.001, device='cpu')
        end_time = time.time()
        training_time_1_3 = end_time - start_time

        plot_training_history(history=history_1_3, save_path='./lesson4/homework/plots/cifar_kernel_comb.png')
    except Exception as e:
        logging.error('An unexpected error occurred')
        raise
    
    
    print(f'____________________\n')
    
    acc1_train = f'{history_3['train_accs'][-1]:.4f}'
    acc2_train = f'{history_5['train_accs'][-1]:.4f}'
    acc3_train = f'{history_7['train_accs'][-1]:.4f}'
    acc4_train = f'{history_1_3['train_accs'][-1]:.4f}'
    
    acc1_test = f'{history_3['test_accs'][-1]:.4f}'
    acc2_test = f'{history_5['test_accs'][-1]:.4f}'
    acc3_test = f'{history_7['test_accs'][-1]:.4f}'
    acc4_test = f'{history_1_3['test_accs'][-1]:.4f}'
    
    loss1_train = f'{history_3['train_losses'][-1]:.4f}'
    loss2_train = f'{history_5['train_losses'][-1]:.4f}'
    loss3_train = f'{history_7['train_losses'][-1]:.4f}'
    loss4_train = f'{history_1_3['train_losses'][-1]:.4f}'
    
    loss1_test = f'{history_3['test_losses'][-1]:.4f}'
    loss2_test = f'{history_5['test_losses'][-1]:.4f}'
    loss3_test = f'{history_7['test_losses'][-1]:.4f}'
    loss4_test = f'{history_1_3['test_losses'][-1]:.4f}'

    # Итоговая таблица
    result_table = {
    'Модель': ['3x3', '5x5', '7x7', '1x1 + 3x3'],
    'Точность на train': [acc1_train, acc2_train, acc3_train, acc4_train],
    'Точность на test': [acc1_test, acc2_test, acc3_test, acc4_test],
    'Loss на train': [loss1_train, loss2_train, loss3_train, loss4_train],
    'Loss на test': [loss1_test, loss2_test, loss3_test, loss4_test],
    'Количество параметров': [parametrs_3, parametrs_5, parametrs_7, parametrs_1_3],
    f'Время обучения ({EPOCHS})': [training_time_3, training_time_5, training_time_7, training_time_1_3]
    }
    
    
    result_json = pd.DataFrame(result_table)
    save_path = './lesson4/homework/results/architecture_analysis/cifar_results.json'
    result_json.to_json(save_path, orient='records', force_ascii=False, indent=4)
    print(result_json)