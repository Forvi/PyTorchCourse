import time
import psutil
import torch
from torchvision import transforms
from utils.datasets import CustomImageDataset
import matplotlib.pyplot as plt

process = psutil.Process()

sizes = [64, 128, 224, 512]
num_images = 100

load_times = []
aug_times = []
mem_usages = []

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor()
])

root = './data/train'

for size in sizes:
    print(f"Размер: {size}x{size}")

    transform = transforms.Compose([
        transforms.Resize((size, size)),
    ])

    dataset = CustomImageDataset(root, transform=transform, target_size=(size, size))

    start_mem = process.memory_info().rss / (1024 ** 2)  # в МБ
    start_time = time.time()
    images = []
    for i in range(num_images):
        img, label = dataset[i]
        images.append(img)
    load_time = time.time() - start_time
    end_mem = process.memory_info().rss / (1024 ** 2)
    mem_usage = end_mem - start_mem

    print(f"Время загрузки: {load_time:.2f} с, Память: {mem_usage:.2f} МБ")

    start_time = time.time()
    augmented_images = [augmentation(img) for img in images]
    aug_time = time.time() - start_time
    print(f"Время аугментаций: {aug_time:.2f} с")

    load_times.append(load_time)
    aug_times.append(aug_time)
    mem_usages.append(mem_usage)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(sizes, load_times, marker='o', label='Время загрузки')
plt.plot(sizes, aug_times, marker='o', label='Время аугментаций')
plt.xlabel('Размер изображения')
plt.ylabel('Время (сек)')
plt.title('Время обработки в зависимости от размера')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(sizes, mem_usages, marker='o', color='red')
plt.xlabel('Размер изображения')
plt.ylabel('Память (МБ)')
plt.title('Потребление памяти в зависимости от размера')
plt.grid(True)

plt.tight_layout()
plt.savefig('./lesson5/homework/images/sizes/size_experiment_results.png')
