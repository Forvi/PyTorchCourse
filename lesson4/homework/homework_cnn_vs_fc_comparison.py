import time
import pandas as pd
import json
from utils.visualization_utils import plot_four_history_cnn_fc
from utils.comparison_utils import get_cifar_loaders, get_mnist_loaders
from models.fc_models import FC
from models.cnn_models import CIFARCNN, CNN
from utils.training_utils import count_parameters, train_model
import logging

if __name__ == "__main__":
    
    EPOCHS = 10
    logging.basicConfig(level=logging.INFO, filename="./lesson4/homework/results/cifar_comparison/cifar_fc_log.log",filemode="w")
    
    # Полносвязная нейронная сеть
    print(f'____________________\n')
    
    try:
        logging.debug('FC IS STARTING...')
        
        model_fc = FC(input_size=3072)
        logging.info('Model initialized')
        
        print(model_fc)
        
        logging.info('Counting the number of parameters')
        parametrs_fc = count_parameters(model_fc)

        logging.info('Getting the dataset')
        
        try:
            train_loader, test_loader = get_cifar_loaders(batch_size=128)
        except:
            logging.exception('Failed to get dataset')

        logging.info('Train model')
        start_time = time.time()
        history_fc = train_model(model_fc, train_loader, test_loader, epochs=EPOCHS, lr=0.001, device='cpu')
        end_time = time.time()
        training_time_fc = end_time - start_time
    except Exception as e:
        logging.error('An unexpected error occurred')
        raise
        

        # Свёрточная нейронная сеть
        print(f'____________________\n')
        
    try:
        logging.basicConfig(level=logging.INFO, filename="./lesson4/homework/results/cifar_comparison/cifar_cnn_log.log",filemode="w")
        logging.debug('CNN IS STARTING...')
        
        # model_cnn = CNN()
        model_cnn = CIFARCNN()
        
        print(model_cnn)
        parametrs_cnn = count_parameters(model_cnn)

        logging.info('Train model')
        start_time = time.time()
        history_cnn = train_model(model_cnn, train_loader, test_loader, epochs=EPOCHS, lr=0.001, device='cpu')
        end_time = time.time()
        training_time_cnn = end_time - start_time

        plot_four_history_cnn_fc(history_cnn=history_cnn, history_fc=history_fc, save_path='./lesson4/homework/plots/fc_vs_cnn_cifar.png')
    except Exception as e:
        logging.error('An unexpected error occurred')
        raise
    
    
    print(f'____________________\n')

    # Итоговая таблица
    result_table = {
    'Модель': ['CNN', 'FC'],
    'Точность на train': [f'{history_cnn['train_accs'][-1]:.4f}', f'{history_fc['train_accs'][-1]:.4f}'],
    'Точность на test': [f'{history_cnn['test_accs'][-1]:.4f}', f'{history_fc['test_accs'][-1]:.4f}'],
    'Loss на train': [f'{history_cnn['train_losses'][-1]:.4f}', f'{history_fc['train_losses'][-1]:.4f}'],
    'Loss на test': [f'{history_cnn['test_losses'][-1]:.4f}', f'{history_fc['test_losses'][-1]:.4f}'],
    'Количество параметров': [parametrs_cnn, parametrs_fc],
    f'Время обучения ({EPOCHS})': [training_time_cnn, training_time_fc]
    }
    
    
    result_json = pd.DataFrame(result_table)
    save_path = './lesson4/homework/results/cifar_comparison/cifar_results.json'
    result_json.to_json(save_path, orient='records', force_ascii=False, indent=4)
    print(result_json)