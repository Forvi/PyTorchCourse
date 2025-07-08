import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
import os

def show_images(images, labels=None, nrow=8, title=None, size=128, image_name='default'):
    images = images[:nrow]
    
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    
    fig, axes = plt.subplots(1, nrow, figsize=(nrow*2, 2))
    if nrow == 1:
        axes = [axes]
    
    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'./lesson5/homework/images/{image_name}.png')


def show_single_augmentation(original_img, augmented_img, title="Аугментация", image_name='single'):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    resize_transform = transforms.Resize((128, 128), antialias=True)

    if not isinstance(original_img, torch.Tensor):
        to_tensor = transforms.ToTensor()
        original_img = to_tensor(original_img)
    if not isinstance(augmented_img, torch.Tensor):
        to_tensor = transforms.ToTensor()
        augmented_img = to_tensor(augmented_img)
        
    orig_resized = resize_transform(original_img)
    aug_resized = resize_transform(augmented_img)
    
    orig_np = orig_resized.permute(1, 2, 0).numpy()
    orig_np = np.clip(orig_np, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title('Оригинал')
    ax1.axis('off')
    
    aug_np = aug_resized.permute(1, 2, 0).numpy()
    aug_np = np.clip(aug_np, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')

    plt.tight_layout()
    
    os.makedirs('./lesson5/homework/images', exist_ok=True)
    plt.savefig(f'./lesson5/homework/images/{image_name}.png')
    plt.close()  
    


def show_multiple_augmentations(original_img, augmented_imgs, titles, image_name='multiple'):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))
    
    resize_transform = transforms.Resize((128, 128), antialias=True)
    
    if not isinstance(original_img, torch.Tensor):
        to_tensor = transforms.ToTensor()
        original_img = to_tensor(original_img)
    
    orig_resized = resize_transform(original_img)
    
    orig_np = orig_resized.permute(1, 2, 0).numpy()
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')
    
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        if not isinstance(aug_img, torch.Tensor):
            aug_img = transforms.ToTensor()(aug_img)
        
        aug_resized = resize_transform(aug_img)
        aug_np = aug_resized.permute(1, 2, 0).numpy()
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    os.makedirs('./lesson5/homework/images', exist_ok=True)
    plt.savefig(f'./lesson5/homework/images/{image_name}.png')
    plt.close() 
