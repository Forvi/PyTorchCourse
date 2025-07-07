import time
import pandas as pd
import json
from models.custom_layers import AllCustomCNN, AttentionCNN, CustomConvCNN, GeMCNN, StandardCNN, SwishCNN
from utils.visualization_utils import plot_training_history
from utils.comparison_utils import get_cifar_loaders, get_mnist_loaders
from models.fc_models import FC
from models.cnn_models import CIFARCNN, CNN, CNN_1x1_3x3, CustomCNN, LayersCNN
from utils.training_utils import count_parameters, train_model
import logging

if __name__ == "__main__":
    EPOCHS = 10
    logging.basicConfig(level=logging.INFO, filename="./lesson4/homework/results/architecture_analysis/custom_layers_analysis.log", filemode="w")
    
    try:
        logging.info('Getting the dataset')
        train_loader, test_loader = get_cifar_loaders(batch_size=128)
    except Exception:
        logging.exception('Failed to get dataset. Exiting.')
        exit(1) # Завершаем выполнение, если не удалось загрузить данные
    
    experiment_models = {
        "StandardCNN": StandardCNN(input_channels=3, num_classes=10),
        "CustomConvCNN": CustomConvCNN(input_channels=3, num_classes=10),
        "AttentionCNN": AttentionCNN(input_channels=3, num_classes=10),
        "SwishCNN": SwishCNN(input_channels=3, num_classes=10),
        "GeMCNN": GeMCNN(input_channels=3, num_classes=10),
        "AllCustomCNN": AllCustomCNN(input_channels=3, num_classes=10),
    }

    all_results = []

    for model_name, model_instance in experiment_models.items():
        logging.info(f'Starting experiment for model: {model_name}')
        print(f"\n--- Model: {model_name} ---")
        print(model_instance)
        
        try:
            params = count_parameters(model_instance)
            logging.info(f'Number of parameters for {model_name}: {params}')

            logging.info(f'Training {model_name}')
            start_time = time.time()
            history = train_model(model_instance, train_loader, test_loader, epochs=EPOCHS, lr=0.001, device='cpu')
            end_time = time.time()
            training_time = end_time - start_time
            logging.info(f'Training time for {model_name}: {training_time:.2f} seconds')

            plot_training_history(history=history, save_path=f'./lesson4/homework/plots/custom_layer_{model_name.lower()}.png')
            
            all_results.append({
                "Модель": model_name,
                "Точность на train": f"{history['train_accs'][-1]:.4f}",
                "Точность на test": f"{history['test_accs'][-1]:.4f}",
                "Loss на train": f"{history['train_losses'][-1]:.4f}",
                "Loss на test": f"{history['test_losses'][-1]:.4f}",
                "Количество параметров": params,
                f"Время обучения ({EPOCHS})": training_time
            })
            
        except Exception as e:
            logging.exception(f'An error occurred during training {model_name}')
            print(f"Error training {model_name}: {e}")
            all_results.append({
                "Модель": model_name,
                "Точность на train": "N/A", "Точность на test": "N/A",
                "Loss на train": "N/A", "Loss на test": "N/A",
                "Количество параметров": params if 'params' in locals() else "N/A",
                f"Время обучения ({EPOCHS})": "N/A"
            })


    print(f'\n{"="*50}\nALL EXPERIMENTS RESULTS\n{"="*50}')
    results_df = pd.DataFrame(all_results)
    print(results_df)

    save_json_path = './lesson4/homework/results/architecture_analysis/custom_layers_results.json'
    results_df.to_json(save_json_path, orient='records', force_ascii=False, indent=4)
    logging.info(f'Results saved to {save_json_path}')