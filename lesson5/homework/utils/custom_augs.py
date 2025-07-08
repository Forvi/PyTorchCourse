import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

class RandomGaussianBlur:
    """Случайное размытие с вероятностью p."""
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max
    
    def __call__(self, img):
        # img — PIL Image
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

class RandomPerspectiveWarp:
    """Случайная перспектива с вероятностью p."""
    def __init__(self, p=0.5, distortion_scale=0.5):
        self.p = p
        self.distortion_scale = distortion_scale
    
    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = TF.get_perspective_params(
                startpoints=[(0,0), (width,0), (width,height), (0,height)],
                distortion_scale=self.distortion_scale,
                height=height,
                width=width
            )
            return TF.perspective(img, startpoints, endpoints)
        return img

class RandomBrightnessContrast:
    """Случайная яркость и контраст с вероятностью p."""
    def __init__(self, p=0.5, brightness_range=(0.7,1.3), contrast_range=(0.7,1.3)):
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, img):
        if random.random() < self.p:
            brightness_factor = random.uniform(*self.brightness_range)
            contrast_factor = random.uniform(*self.contrast_range)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
            return img
        return img
