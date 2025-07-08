from utils.custom_augs import RandomBrightnessContrast, RandomGaussianBlur, RandomPerspectiveWarp
from utils.datasets import CustomImageDataset
from utils.utils import show_single_augmentation
from utils.extra_augs import AddGaussianNoise, RandomErasingCustom, CutOut
import torch
import torch.nn.functional as F


if __name__ == '__main__':
    
    root = './data/train'
    dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
    original_img, label = dataset[0]

    custom_augs = [
        (RandomGaussianBlur(p=1.0), "Random Gaussian Blur"),
        (RandomPerspectiveWarp(p=1.0), "Random Perspective Warp"),
        (RandomBrightnessContrast(p=1.0), "Random Brightness/Contrast"),
    ]

    extra_augs = [
        (AddGaussianNoise(0., 0.2), "Add Gaussian Noise"),
        (RandomErasingCustom(p=1.0), "Random Erasing Custom"),
        (CutOut(p=1.0, size=(32,32)), "CutOut"),
    ]

    for aug, name in custom_augs:
        aug_img = aug(original_img)
        show_single_augmentation(original_img, aug_img, title=name, image_name=name.replace(" ", "_").lower())

    to_tensor = torch.transforms.ToTensor()
    for aug, name in extra_augs:
        # Применяем к тензору, поэтому сначала преобразуем оригинал
        tensor_img = to_tensor(original_img)
        aug_img = aug(tensor_img)
        # Для визуализации преобразуем обратно в PIL
        aug_img_pil = F.to_pil_image(aug_img)
        show_single_augmentation(original_img, aug_img_pil, title=name, image_name=name.replace(" ", "_").lower())
