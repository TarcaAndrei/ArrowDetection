from __future__ import annotations
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils.state import DataKey, LabelType
from torchvision.transforms import v2
import json
from pathlib import Path


class ArrowDataset(Dataset):
    """
    Dataset class for a generic CSV dataset.
    """

    def __init__(self, annotation_file: str, root_dir_json: str, classes: list,
                 transforms: v2.Compose, data_type: str = "train", subset_dim: int | None = None):
        """
        Initializes the dataset for object detection of non-arrow objects.

        Args:
            annotation_file: Full path to the .csv file listing all the JSON files.
            classes: list of labels corresponding to the classes.
            sep: Separator for CSV file.
            data_type: Dataset type, one of ('train', 'val', 'test').
            transforms: Transform applied to the image; if None, standard normalization is applied.
            subset_dim: If set to a number, uses a subset of the entire dataset.
        """
        super().__init__()
        all_data = pd.read_csv(annotation_file)
        self.transforms = transforms
        self.classes = classes
        self.root_dir_json = Path(root_dir_json)

        if data_type not in ('train', 'val', 'test'):
            raise ValueError("Dataset type is not one of train, val, test!")
        all_entities = all_data[all_data["type"]
                                == data_type].reset_index()
        if subset_dim is not None:
            all_entities = all_entities[:subset_dim]
        self.data = all_entities
        self._iter_index = 0

    def __getitem__(self, idx: int):
        """
        Retrieves a sample from a specific index in the dataset.

        Args:
            idx: The index from the .csv file.
        """
        if idx >= len(self):
            raise IndexError("Dataset out of bound")

        json_label = self.data.loc[idx].at["json_label"]
        img_path = self.data.loc[idx].at["image_path"]

        metadata = {
            'img_path': img_path,
            'json_label': json_label,
        }

        image = self._get_image(img_path)
        bounding_boxes = self._get_bounding_box_objects(json_label)

        image_transformed, bbox_transformed = self.transform(
            image, bounding_boxes)

        return {
            DataKey.IMAGE: image_transformed,
            DataKey.LABEL: {
                LabelType.BBOX: bbox_transformed
            },
            DataKey.METADATA: metadata
        }

    def _get_bounding_box_objects(self, json_file: str) -> dict[str, list]:
        """
        Extracts bounding box objects from the JSON content.

        Args:
            data: A dictionary containing all the information from the JSON file.

        Returns:
            list[dict[str, list[int] | str]]:
                A list of dictionaries for each object, containing:
                    - **bbox**: A list of four integers [center_x, center_y, width, height].
                    - **class**: A string representing the class name of the object.
        """
        with open(self.root_dir_json / json_file, "r") as fisier:
            data = json.load(fisier)
            labels = []
            boxes = []
            for obj in data:
                clasa_obj = obj['class'].lower()
                if clasa_obj not in self.classes:
                    continue
                labels.append(self.classes.index(clasa_obj))
                [center_x, center_y, width, height] = obj['bbox']
                boxes.append([center_x, center_y, width, height])

            return {
                "labels": labels,
                "boxes": boxes
            }

    def _get_image(self, img_path: str) -> np.ndarray:
        """
        Retrieves and processes the image referenced in the JSON file.
        Args:
            img_path: Relative path to the image.

        Returns:
            torch.Tensor: A tensor representing the normalized image in C, H, W format.
        """
        bgr_img = cv2.imread(str(self.base_dir / img_path), -1)
        image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return image

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
                (:class:`torch.Tensor`): A tensor representing the image.
                (list[dict[str, list[int] | str]]): A list of dictionaries for each object.
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
