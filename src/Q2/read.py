import os
import re
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional


class Q2Dataset(Dataset):
    def __init__(self, data_dir, transform: Optional[transforms.Compose] = None, train: Optional[bool] = True) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        if os.path.exists(self.data_dir):
            files = os.listdir(self.data_dir)
            file_absolute_paths = []
            labels = []
            for file in files:
                labels.append(re.split(r"\d+", file)[0])
                file_absolute_path = self.data_dir + file
                file_absolute_paths.append(file_absolute_path)
            self.data = pd.get_dummies(labels, dtype=int)
            self.data["file"] = file_absolute_paths
        else:
            raise FileNotFoundError
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = Image.open(self.data.iloc[index, -1]).convert("RGB")
        label = self.data.iloc[index, :-1]
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform is None:
            if self.train:
                self.transform = transforms.Compose([
                    transforms.RandomRotation(45),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.Resize((224, 224), interpolation=Image.BICUBIC), 
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean and variance in ImageNet
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224), interpolation=Image.BICUBIC), 
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean and variance in ImageNet
                ])
        
        img = self.transform(image)
        return img, label
