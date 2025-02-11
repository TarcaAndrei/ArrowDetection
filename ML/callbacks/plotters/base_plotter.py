from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import torch

from callbacks.base_callbacks import CallbackWithOutput
from utils import DataKey, EvaluatorState, StoreKey, TrainerState


class ImagePlotter(CallbackWithOutput, ABC):
    def __init__(self, label_decoder: dict[int, str], **kwargs):
        """
        Callback for plotting images during training

        Args:
            label_decoder: dict mapping class ids to their original names
        """
        super().__init__(**kwargs)
        self.output_dir = Path(self.output_dir) / 'images'
        self.label_decoder = label_decoder

    def _get_image_names(self, store):
        """
        Returns a list of names for each image in the batch
        """
        return [
            "--".join(Path(metadata['img_path']).parts[-2:]) for metadata in store[StoreKey.DATA][DataKey.METADATA]
        ]

    @abstractmethod
    def _draw_images(self, store):
        """
        Plots the labels or predictions (depending on availability) and returns a list of images  
        """

    def _plot_images(self, state, stage):
        """
        Plots the images in a batch.
        """
        output_path = self.output_dir / stage \
            / f"epoch_{str(state.epoch).zfill(6)}" \
            / f"iter_{str(state.iteration).zfill(6)}"
        os.makedirs(output_path, exist_ok=True)

        names = self._get_image_names(state.store)
        images = self._draw_images(state.store)

        for i, (name, image) in enumerate(zip(names, images)):
            r, g, b = list(image)
            cv2.imwrite(
                str(output_path / f"{i}_{name}"), torch.stack([b, g, r], axis=-1).numpy())

    def on_train_batch_end(self, state: TrainerState) -> None:
        self._plot_images(state, 'train')

    def on_evaluation_batch_end(self, state: EvaluatorState) -> None:
        self._plot_images(state, 'eval')
