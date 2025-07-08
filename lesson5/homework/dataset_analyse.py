import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

root_dir = './data/train'
class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
class_counts = []
widths, heights = [], []

for class_name in class_names:
    class_dir = os.path.join(root_dir, class_name)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    class_counts.append(len(images))
    for img_name in images:
        img_path = os.path.join(class_dir, img_name)
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

min_size = (min(widths), min(heights))
max_size = (max(widths), max(heights))
mean_size = (np.mean(widths), np.mean(heights))

plt.scatter(widths, heights, alpha=0.5, s=10)
plt.title('Распределение размеров изображений (ширина vs высота)')
plt.xlabel('Ширина')
plt.ylabel('Высота')
plt.grid(True)
plt.savefig(f'./lesson5/homework/images/analyse/size_image_scatter.png')

plt.bar(class_names, class_counts)
plt.title('Количество изображений по классам')
plt.xlabel('Класс')
plt.ylabel('Количество изображений')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig(f'./lesson5/homework/images/analyse/count_image_hist.png')
