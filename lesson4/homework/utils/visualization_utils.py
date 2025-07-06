import matplotlib.pyplot as plt
import os


def plot_training_history(history, save_path):
    """Сохраняет график с историей обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_accs'], label='Train Accuracy')
    ax2.plot(history['test_accs'], label='Test Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    
def plot_four_history_cnn_fc(history_cnn, history_fc, save_path):
    """Сохраняет график с историей обучения 2 моделей"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1.plot(history_cnn['train_losses'], label='Train Loss')
    ax1.plot(history_cnn['test_losses'], label='Test Loss')
    ax1.set_title('Loss CNN')
    ax1.legend()
    
    ax2.plot(history_cnn['train_accs'], label='Train Accuracy')
    ax2.plot(history_cnn['test_accs'], label='Test Accuracy')
    ax2.set_title('Accuracy CNN')
    ax2.legend()
    
    ax3.plot(history_fc['train_losses'], label='Train Loss')
    ax3.plot(history_fc['test_losses'], label='Test Loss')
    ax3.set_title('Loss FC')
    ax3.legend()
    
    ax4.plot(history_fc['train_accs'], label='Train Accuracy')
    ax4.plot(history_fc['test_accs'], label='Test Accuracy')
    ax4.set_title('Accuracy FC')
    ax4.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)