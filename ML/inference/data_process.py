from torchvision.transforms import v2
import numpy as np
import torch
import cv2


class PreprocessData:
    def __init__(self, device) -> None:
        DEFAULT_MEAN = [0.485, 0.456, 0.406]
        DEFAULT_STD = [0.229, 0.224, 0.225]
        self.transform_imgs = v2.Compose([
            v2.ToImage(),
            # v2.Grayscale(num_output_channels=3),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            v2.Normalize(mean=DEFAULT_MEAN,
                         std=DEFAULT_STD),
        ])
        self.device = device

    def preprocess_image(self, image_path: str):
        initial_image = cv2.imread(image_path, -1)
        if initial_image.shape[-1] == 4:
            # print("4 channels")
            image = cv2.cvtColor(initial_image, cv2.COLOR_BGRA2RGB)
        else:
            # print("3 channels")
            image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)
        return self.transform_imgs(image).to(device=self.device)
