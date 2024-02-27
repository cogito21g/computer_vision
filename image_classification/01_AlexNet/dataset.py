import torch
from torch.utils.data import Dataset
from torchvision import models, transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = None
        self.label = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.data)
