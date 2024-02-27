import argparse
import torch
from PIL import Image
from torchvision import models, transforms
from torchinfo import summary

def parsing():
    args = argparse.ArgumentParser()
    args.add_argument("--verbose", default=1, help="show status", type=str)
    args.add_argument("--save_dir", default="result", help="model states directory", type=str)
    
    return args.parse_args()




if __name__=="__main__":
    # 모델 불러오기
    model = models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
    summary(model, (1, 3, 224, 224), device="cpu")

    # 클래스 정보 파일 불러오기
    with open("../data/imagenet_classes.txt", "r") as file:
        classes = file.read().splitlines()

    print(f"class num: {len(classes)}")


    # 데이터 전처리
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            ),
        ]
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1").eval().to(device)

    tensors = []
    files = ["./data/images/airplane.jpg", "./data/images/bus.jpg"]
    for file in files:
        image = Image.open(file)
        tensors.append(transform(image))

    tensors = torch.stack(tensors)
    print(f"입력 텐서의 크기: {tensors.shape}")