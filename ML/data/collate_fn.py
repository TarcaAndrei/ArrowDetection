from __future__ import annotations

import torch

from utils.state import DataKey


class CollateFn:
    """
    A unified collate function for creating data loaders for the datasets.
    """

    device: torch.device

    def __init__(self, device: torch.device) -> None:
        """
        Args:
            device: The device (e.g., 'cpu' or 'cuda') on which tensors should be placed.
        """
        self.device = device

    def __call__(self,
                 batch: list[dict[DataKey, torch.Tensor | dict]]
                 ) -> dict[DataKey, torch.Tensor | dict[str, torch.Tensor] | list]:
        """
        Process a batch of data samples into tensors for model training.

        Args:
            batch: List of dictionaries containing image, label, and metadata.

        Returns:
            A dictionary containing processed images, labels, and metadata.
        """
        images: list[torch.Tensor] = []
        final_metadata: list[dict[str, str]] = []
        final_labels: list[dict[str, torch.Tensor]] = []

        for sample in batch:
            image, ground_truth, metadata = (
                sample[DataKey.IMAGE].to(self.device),  # type: ignore[union-attr]
                sample[DataKey.LABEL],
                sample[DataKey.METADATA]
            )

            # Flatten labels dict (remove LabelType layer) and convert to tensor, where:
            # ground_truth = {
            #     LabelType.TYPE: {
            #         label_name: value
            #         ...
            #     }
            #     ...
            # }
            current_labels = {
                label_name: torch.Tensor(value).to(self.device)
                for labels in ground_truth.values()
                for label_name, value in labels.items()
            }

            # Append the processed image, labels, and metadata for the current sample to their respective lists.
            images.append(image)  # type: ignore[arg-type]
            final_labels.append(current_labels)  # type: ignore[arg-type]
            final_metadata.append(metadata)  # type: ignore[arg-type]

        return {
            DataKey.IMAGE: torch.stack(images, dim=0).to(self.device),
            DataKey.LABEL: final_labels,
            DataKey.METADATA: final_metadata
        }
