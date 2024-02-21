# Dataset

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, root, train, transform=None, target_transform=None):
        super(SegmentationDataset, self).__init__()
        self.root = os.path.join(root, "VOCdevkit", "VOC2012")
        file_type = "train" if train else "val"
        file_path = os.path.join(
            self.root, "ImageSets", "Segmentation", f"{file_type}.txt"
        )
        with open(os.path.join(self.root, "classes.json"), "r") as file:
            self.categories = json.load(file)
        self.files = open(file_path).read().splitlines()