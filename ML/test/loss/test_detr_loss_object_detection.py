# pylint: disable=W0621
from unittest.mock import MagicMock

import pytest
import torch

from models import DETRLoss


@pytest.fixture
def num_classes() -> int:
    """Fixture to define the number of classes for the object detection model.

    Returns:
        int: Number of classes in the dataset (2 in this case).
    """
    return 2


@pytest.fixture
def matcher() -> MagicMock:
    """Fixture to mock the matcher that returns predefined indices for testing.

    Returns:
        MagicMock: Mock object that mimics the behavior of a matcher.
    """
    # Mock matcher to return predefined indices for testing
    mock = MagicMock()
    mock.return_value = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    return mock


@pytest.fixture
def weight_dict() -> dict[str, float]:
    """Fixture to define the loss weights for the criterion.

    Returns:
        dict: A dictionary containing the weights for different loss components.
    """
    return {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0, 'loss_points': 2.0}


@pytest.fixture
def eos_coef() -> float:
    """Fixture to define the end-of-sequence coefficient.

    Returns:
        float: The coefficient for handling 'no object' predictions.
    """
    return 0.1


@pytest.fixture
def loss_types() -> list[str]:
    """Fixture to define the types of losses to be calculated.

    Returns:
        list: A list containing the names of the loss components to be computed.
    """
    return ['labels', 'cardinality', 'boxes', 'points']


@pytest.fixture
def criterion(
    num_classes: int,
    matcher: MagicMock,
    weight_dict: dict,
    eos_coef: float,
    loss_types: list[str]
) -> torch.nn.Module:
    """Fixture to create an instance of the DETRLoss class.

    Args:
        num_classes (int): Number of classes in the dataset.
        matcher (MagicMock): Mock matcher object.
        weight_dict (dict): Dictionary containing loss weights.
        eos_coef (float): End-of-sequence coefficient.
        losses (list): List of loss components to compute.

    Returns:
        DETRLoss: Instance of the DETRLoss class.
    """
    return DETRLoss(num_classes, matcher, weight_dict, eos_coef, loss_types)


@pytest.fixture
def outputs() -> dict[str, torch. Tensor]:
    """Fixture to provide example outputs from a model for testing.

    Returns:
        dict: A dictionary containing predicted logits and bounding boxes.
    """
    return {
        # batch size x num_queries x num_classes
        'pred_logits': torch.tensor([[[0.5, 0.5, 10.0], [0.5, 0.5, 0.1]]]),
        # batch size x num_queries x 4
        'pred_boxes': torch.tensor([[[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]]]),
    }


@pytest.fixture
def targets() -> list[dict[str, torch.Tensor]]:
    """Fixture to provide example targets for testing.

    Returns:
        list: A list of dictionaries containing ground truth labels and bounding boxes.
    """
    return [
        {'labels': torch.tensor([2, 1]), 'boxes': torch.tensor(
            [[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]])}
    ]


@pytest.fixture
def indices() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Fixture to provide example matched indices for testing.

    Returns:
        list: A list of tuples containing matched indices for predicted and target boxes.
    """
    return [(torch.tensor([0, 1]), torch.tensor([0, 1]))]


def test_loss_labels(
        criterion: DETRLoss,
        outputs: dict[str, torch. Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Test the label loss computation in the criterion.

    Args:
        criterion (DETRLoss): The criterion object to compute losses.
        outputs (dict): Model outputs containing predicted logits and bounding boxes.
        targets (list): Ground truth targets.
        indices (list): Matched indices for predicted and target boxes.
    """
    num_boxes = 2  # Example number of boxes
    losses = criterion.loss_labels(outputs, targets, indices, num_boxes)
    assert 'loss_ce' in losses
    assert losses['loss_ce'] > 0


def test_loss_boxes(
        criterion: DETRLoss,
        outputs: dict[str, torch. Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Test the bounding box and generalized IoU loss computation in the criterion.

    Args:
        criterion (DETRLoss): The criterion object to compute losses.
        outputs (dict): Model outputs containing predicted logits and bounding boxes.
        targets (list): Ground truth targets.
        indices (list): Matched indices for predicted and target boxes.
    """
    num_boxes = 2
    losses = criterion.loss_boxes(outputs, targets, indices, num_boxes)
    assert 'loss_bbox' in losses
    assert 'loss_giou' in losses
    assert losses['loss_bbox'] >= 0
    assert losses['loss_giou'] >= 0


def test_forward(
    criterion: DETRLoss,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]]
) -> None:
    """Test the overall forward pass of the criterion for loss computation.

    Args:
        criterion (DETRLoss): The criterion object to compute losses.
        outputs (dict): Model outputs containing predicted logits and bounding boxes.
        targets (list): Ground truth targets.
    """
    losses = criterion(outputs, targets)
    assert 'loss_ce' in losses
    assert 'loss_bbox' in losses
    assert 'loss_giou' in losses
    assert losses['loss_ce'] > 0
    assert losses['loss_bbox'] >= 0
    assert losses['loss_giou'] >= 0
