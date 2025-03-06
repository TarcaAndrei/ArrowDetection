from __future__ import annotations
from torchvision.transforms.v2 import Compose, Pad, ToImage, ToDtype, Normalize, Grayscale
from torch.utils.data import Dataset
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
from utils.state import DataKey, LabelType


class CeymoDataset(Dataset):
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

        xml_label = self.data.loc[idx].at["label_file"]
        split = self.data.loc[idx].at["stage"]
        image_name = xml_label.strip().split(".")[0] + ".jpg"
        if split == "validation":
            full_path = self.root_dir / "test"
        else:
            full_path = self.root_dir / "train"
        metadata = {
            'img_path': image_name,
            'json_label': xml_label,
        }
        image, height, width = self._get_image(full_path, image_name)
        bounding_boxes = self._get_bounding_boxes(full_path, xml_label, height, width)
        return self.transforms({
            DataKey.IMAGE: image,
            DataKey.LABEL: {
                LabelType.BBOX: bounding_boxes
            },
            DataKey.METADATA: metadata
        })
    
    def _get_bounding_boxes(self, root_dir, xml_label, height_img, width_img):
        full_path = root_dir / "labels" / xml_label
        labels = []
        boxes = []
        tree = ET.parse(full_path)  
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name in self.class_mapping:
                labels.append(self.classes.index(self.class_mapping[name]))
                bndbox = obj.find("bndbox")
                min_x = int(bndbox.find("xmin").text)
                min_y = int(bndbox.find("ymin").text)
                max_x = int(bndbox.find("xmax").text)
                max_y = int(bndbox.find("ymax").text)
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
