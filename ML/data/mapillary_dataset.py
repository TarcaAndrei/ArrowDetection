from __future__ import annotations
from torchvision.transforms.v2 import Compose, Pad, RandomResizedCrop, ColorJitter, ToImage, ToDtype, Normalize, Grayscale
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import json
import cv2
from utils.state import DataKey, LabelType


class MapillaryDataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, transforms, class_mapping, classes, folders: tuple=("train", "val"), subset_dim: int | None = None) -> None:
        super().__init__()
        all_data = pd.read_csv(csv_file, sep=",")
        all_entities = all_data[all_data["stage"].isin(folders)].reset_index()
        if subset_dim is not None:
            all_entities = all_entities[:subset_dim]
        self.data = all_entities
        self._iter_index = 0
        self.transforms = transforms
        self.root_dir = root_dir
        self.class_mapping = class_mapping
        self.classes = classes
        if isinstance(self.root_dir, str):
            self.root_dir = Path(self.root_dir)

    def __getitem__(self, idx: int):
        """
        Retrieves a sample from a specific index in the dataset.

        Args:
            idx: The index from the .csv file.
        """
        if idx >= len(self):
            raise StopIteration("Dataset out of bound")

        json_label = self.data.loc[idx].at["label_file"]
        split = self.data.loc[idx].at["stage"]
        image_name = json_label.strip().split(".")[0] + ".jpg"
        if split == "validation":
            full_path = self.root_dir / "validation"
        else:
            full_path = self.root_dir / "training"
        metadata = {
            'img_path': image_name,
            'json_label': json_label,
        }
        image, height, width = self._get_image(full_path, image_name)
        bounding_boxes = self._get_bounding_boxes(full_path, json_label, height, width)
        return self.transforms({
            DataKey.IMAGE: image,
            DataKey.LABEL: {
                LabelType.BBOX: bounding_boxes
            },
            DataKey.METADATA: metadata
        })
    
    def _get_bounding_boxes(self, root_dir, json_label, height_img, width_img):
        full_path = root_dir / "v2.0" / "polygons" / json_label
        labels = []
        boxes = []
        with open(full_path, "r") as json_file:
            information = json.load(json_file)
            objects = information["objects"]
            for object in objects:
                if object["label"] in self.class_mapping:
                    labels.append(self.classes.index(self.class_mapping[object["label"]]))
                    polygon = object["polygon"]
                    xs = [point[0] for point in polygon]
                    ys = [point[1] for point in polygon]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    width = max_x - min_x
                    height = max_y - min_y
                    boxes.append((center_x, center_y, width, height))
        return {
            "labels": labels,
            "boxes": boxes
        }


    def _get_image(self, root_dir, image_name):
        full_path = root_dir / "images" / image_name
        img =  cv2.imread(full_path, -1)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        mod_height = 14 - height % 14
        mod_width = 14 - width % 14
        transformation = Compose([
            ToImage(),
            Pad(padding=[0, 0, mod_width, mod_height])
            ])
        final_img = transformation(image)
        height = final_img.shape[1]
        width = final_img.shape[2]
        return final_img, height, width

    def __iter__(self):
        """
        Returns an iterator over the dataset.
        """
        self._iter_index = 0
        return self

    def __next__(self):
        """
        Provides the next item in the dataset during iteration.

        Returns:
            tuple:
                * **image** (:class:`torch.Tensor`): A tensor representing the image.
                * **labels** (list[dict[str, list[int] | str]]): A list of dictionaries for each object.
        """
        if self._iter_index >= len(self):
            raise StopIteration
        item = self[self._iter_index]
        self._iter_index += 1
        return item

    def __len__(self) -> int:
        """
        Returns:
            int: The number of elements in the dataset.
        """
        return len(self.data)
