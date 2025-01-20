import numpy as np
import config
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


transform_both = A.Compose([A.Resize(256, 256),
                            A.HorizontalFlip(p=0.5),
                            ], additional_targets={'image0': 'image'})

transform_only_input = A.Compose([#A.ColorJitter(),
                                  A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
                                  ToTensorV2(),
                                  ])

transform_only_target = A.Compose([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
                                   ToTensorV2(),
                                   ])


class RESIDEDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.list_images = os.listdir(self.input_dir)

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, index):
        input_image_file = self.list_images[index]
        input_image_path = os.path.join(self.input_dir, input_image_file)
        input_image = np.array(Image.open(input_image_path).convert("RGB"))

        target_image_number = input_image_path.split("hazy")[1].split("_")[0]
        target_image_path = self.target_dir + target_image_number + ".png"
        target_image = np.array(Image.open(target_image_path).convert("RGB"))

        # Apply the same transformation to both input and target images
        augmented = transform_both(image=input_image, image0=target_image)
        input_image = augmented['image']
        target_image = augmented['image0']

        # Apply additional transformations
        input_image = transform_only_input(image=input_image)['image']
        target_image = transform_only_target(image=target_image)['image']

        return input_image, target_image

if __name__ == "__main__":
    input_dir = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/RESIDE/hazy"
    target_dir = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/RESIDE/clear"

    dataset = RESIDEDataset(input_dir, target_dir)
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        # save_image(x, "x.png")
        # save_image(y, "y.png")
        import sys

        sys.exit()
