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


transform_both = A.Compose([A.Resize(512, 512),
                            A.HorizontalFlip(p=0.5),
                            ], additional_targets={'image0': 'image'})

transform_only_input = A.Compose([#A.ColorJitter(),
                                  A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
                                  ToTensorV2(),
                                  ])

transform_only_target = A.Compose([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
                                   ToTensorV2(),
                                   ])

class NHHAZEDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_images = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('_hazy.png')]) 
        self.target_images = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('_GT.png')])

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, index):
        input_image = np.array(Image.open(self.input_images[index]))
        target_image = np.array(Image.open(self.target_images[index]))

        # Apply the same transformation to both input and target images
        augmented = transform_both(image=input_image, image0=target_image)
        input_image = augmented['image']
        target_image = augmented['image0']

        # Apply additional transformations
        input_image = transform_only_input(image=input_image)['image']
        target_image = transform_only_target(image=target_image)['image']

        return input_image, target_image
    

if __name__ == "__main__":
    root_dir = "C:/Users/aseem/Downloads/UCSD/ECE 285/Project/NH-HAZE"

    dataset = NHHAZEDataset(root_dir)
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()