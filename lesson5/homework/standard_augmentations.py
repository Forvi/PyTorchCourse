from torchvision import transforms
from utils.datasets import CustomImageDataset
from utils.utils import show_images

root = './data/train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

images, labels = [], []
seen_classes = set()
for img, label in dataset:
    if label not in seen_classes:
        images.append(img)
        labels.append(class_names[label])
        seen_classes.add(label)
    if len(seen_classes) == 5:
        break

to_tensor = transforms.ToTensor()
original_imgs_tensor = [to_tensor(img) for img in images]
show_images(original_imgs_tensor, labels=labels, nrow=5, title="Оригинальные изображения", image_name='original_imgs')

augs = [
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
    ("RandomCrop", transforms.RandomCrop(200, padding=20)),
    ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
    ("RandomRotation", transforms.RandomRotation(degrees=30)),
    ("RandomGrayscale", transforms.RandomGrayscale(p=1.0))
]

for aug_name, aug in augs:
    aug_transform = transforms.Compose([aug, transforms.ToTensor()])
    aug_imgs = [aug_transform(img) for img in images]
    show_images(aug_imgs, labels=labels, nrow=5, title=f"Аугментация: {aug_name}", image_name='aug_imgs')

all_aug_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(200, padding=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(degrees=20),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()
])
all_aug_imgs = [all_aug_pipeline(img) for img in images]
show_images(all_aug_imgs, labels=labels, nrow=5, title="Все аугментации вместе", image_name='all_aug')
