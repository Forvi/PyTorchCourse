from torchvision import transforms
from utils.extra_augs import AddGaussianNoise, AugmentationPipeline, RandomErasingCustom, CutOut
import os
from utils.datasets import CustomImageDataset
from utils.utils import show_single_augmentation


# light
light_pipeline = AugmentationPipeline()
light_pipeline.add_augmentation('horizontal_flip', transforms.RandomHorizontalFlip(p=0.5))
light_pipeline.add_augmentation('gaussian_noise', AddGaussianNoise(0., 0.1))

# medium
medium_pipeline = AugmentationPipeline()
medium_pipeline.add_augmentation('horizontal_flip', transforms.RandomHorizontalFlip(p=0.5))
medium_pipeline.add_augmentation('random_crop', transforms.RandomCrop(200, padding=20))
medium_pipeline.add_augmentation('color_jitter', transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
medium_pipeline.add_augmentation('random_erasing', RandomErasingCustom(p=0.5))

# heavy
heavy_pipeline = AugmentationPipeline()
heavy_pipeline.add_augmentation('horizontal_flip', transforms.RandomHorizontalFlip(p=1.0))
heavy_pipeline.add_augmentation('random_crop', transforms.RandomCrop(180, padding=30))
heavy_pipeline.add_augmentation('color_jitter', transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1))
heavy_pipeline.add_augmentation('random_rotation', transforms.RandomRotation(degrees=30))
heavy_pipeline.add_augmentation('cutout', CutOut(p=0.7, size=(32, 32)))
heavy_pipeline.add_augmentation('gaussian_noise', AddGaussianNoise(0., 0.2))

root = './data/train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
os.makedirs('./lesson5/homework/images/augmentation_pipeline', exist_ok=True)

images = [dataset[i][0] for i in range(5)]

pipelines = {
    'light': light_pipeline,
    'medium': medium_pipeline,
    'heavy': heavy_pipeline,
}

for name, pipeline in pipelines.items():
    for idx, img in enumerate(images):
        augmented_img = pipeline.apply(img)
        show_single_augmentation(img, augmented_img, title=f'{name} augmentation', image_name=f'pipeline_{name}_img{idx}')
