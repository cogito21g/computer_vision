import torch
from torchvision import models
from torchinfo import summary

model = models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
summary(model, (1, 3, 224, 224), device="cpu")