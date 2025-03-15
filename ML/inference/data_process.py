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
        height, width = image.shape[:2]
        image = self.pre_transform(image)
        mod_height = 14 - height % 14
        mod_width = 14 - width % 14
        transformation = v2.Compose([
            v2.Pad(padding=[0, 0, mod_width, mod_height])
            ])
        image = transformation(image)
        # image = F.adjust_contrast(image, 1.0)
        return self.post_transform(image).to(device=self.device)

    def preprocess_batch_images(self, root_folder, image_list, start_height=0, end_height=812, start_width=0, end_width=1750):
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
            image = image[start_height:end_height, start_width:end_width]
            all_images.append(image)
            initial_images.append(image)
        height, width = all_images[0].shape[:2]
        mod_height = 14 - height % 14
        if mod_height == 14:
            mod_height = 0
        mod_width = 14 - width % 14
        if mod_width == 14:
            mod_width = 0

        all_images = self.pre_transform(all_images)
        transformation = v2.Compose([
            v2.Pad(padding=[0, 0, mod_width, mod_height])
            ])
        all_images = transformation(all_images)
        return initial_images, torch.stack(self.post_transform(all_images)).to(device=self.device)

    def preprocess_sequences(self, image_list, end_height=812, end_width=1750):
        all_images = []
        initial_images = []
        for image in image_list:
            image = image[0:end_height, 0:end_width]
            all_images.append(image)
            initial_images.append(image)
        height, width = all_images[0].shape[:2]
        mod_height = 14 - height % 14
        if mod_height == 14:
            mod_height = 0
        mod_width = 14 - width % 14
        if mod_width == 14:
            mod_width = 0

        all_images = self.pre_transform(all_images)
        transformation = v2.Compose([
            v2.Pad(padding=[0, 0, mod_width, mod_height])
            ])
        all_images = transformation(all_images)
        return initial_images, torch.stack(self.post_transform(all_images)).to(device=self.device)
