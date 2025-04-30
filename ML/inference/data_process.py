from torchvision.transforms import v2
import torchvision.transforms.functional as F
import numpy as np
import torch
import cv2

import os

class PreprocessData:
    def __init__(self, device) -> None:
        DEFAULT_MEAN = [0.485, 0.456, 0.406]
        DEFAULT_STD = [0.229, 0.224, 0.225]
        self.pre_transform = v2.Compose([
            v2.ToImage(),
        ])
        self.post_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
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
        initial_image = image
        height, width = image.shape[:2]
        image = self.pre_transform(image)
        transformation = v2.Compose([
            v2.Resize((518, 1666))
            ])
        image = transformation(image)
        # image = F.adjust_contrast(image, 1.0)
        return initial_image, self.post_transform(image).to(device=self.device)

    def preprocess_batch_images(self, root_folder, image_list):
        all_images = []
        initial_images = []
        for image_path in image_list:
            initial_image = cv2.imread(os.path.join(root_folder, image_path), -1)
            if initial_image.shape[-1] == 4:
                # print("4 channels")
                image = cv2.cvtColor(initial_image, cv2.COLOR_BGRA2RGB)
            else:
                # print("3 channels")
                image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)
            all_images.append(image)
            initial_images.append(image)
        all_images = self.pre_transform(all_images)
        transformation = v2.Compose([
            v2.Resize((518, 1666))
            ])
        all_images = transformation(all_images)
        return initial_images, torch.stack(self.post_transform(all_images)).to(device=self.device)

    def preprocess_sequences(self, image_list):
        all_images = []
        initial_images = []
        for image in image_list:
            all_images.append(image)
            initial_images.append(image)

        all_images = self.pre_transform(all_images)
        transformation = v2.Compose([
            v2.Resize((518, 1666))
            ])
        all_images = transformation(all_images)
        return initial_images, torch.stack(self.post_transform(all_images)).to(device=self.device)
