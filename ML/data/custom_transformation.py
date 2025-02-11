import numpy as np
import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms import v2


class ObjectDetectionTransformation(nn.Module):

    def __init__(self, final_img_size: tuple = (512, 1664)) -> None:
        """
        Args
            final_img_size: the final image dims to which to padd the initial img
        """
        super().__init__()
        self.final_img_size = final_img_size

    def _pad_image(self, initial_image_dims: tuple[int, int]) -> list[int]:
        """
        Function to calculate the padding for an image shape to obtain the desired image shape
        the desired image shape must be greather or equal to the actual image shape
        Args:
            initial_image: height and width of the initial image
        Returns:
            a list of 4 number representing the padding in the form:
                [left, top, right, bottom] padding
        """
        h, w = initial_image_dims
        if (h, w) == self.final_img_size:
            return [0, 0, 0, 0]
        w_padd = self.final_img_size[1] - w
        h_padd = self.final_img_size[0] - h
        assert w_padd >= 0 and h_padd >= 0
        l_pad = w_padd // 2
        r_pad = w_padd // 2 + w_padd % 2
        t_pad = h_padd // 2
        b_pad = h_padd // 2 + h_padd % 2
        return [l_pad, t_pad, r_pad, b_pad]

    def forward(self, img: np.ndarray, bboxes: list[dict]) -> tuple[torch.Tensor, list[dict]]:
        """
        Args
            img: the image on which to apply the transformations
            bboxes: a list of dicts that contain the "class" and "bbox" annotations for each object
        Returns
            the image and the labels transformed 
        """
        h, w = img.shape[:2]
        labels = [b["class"] for b in bboxes]
        coordinates = [b["bbox"] for b in bboxes]
        coordinates_tensor = tv_tensors.BoundingBoxes(
            coordinates, format="CXCYwH", canvas_size=(h, w))
        padding = self._pad_image(img.shape[:2])
        DEFAULT_MEAN = [0.485, 0.456, 0.406]
        DEFAULT_STD = [0.229, 0.224, 0.225]

        STANDARD_TRANSFORMATION = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=DEFAULT_MEAN,
                         std=DEFAULT_STD)])
        transform = v2.Compose([
            STANDARD_TRANSFORMATION,
            v2.Pad(padding)
        ])
        img_transformed, coordinates_transformed = transform(
            img, coordinates_tensor)
        final_labels = []
        for (lbl, coord) in zip(labels, coordinates_transformed):
            final_labels.append({
                "class": lbl,
                "bbox": coord.tolist()
            })
        return img_transformed, final_labels
