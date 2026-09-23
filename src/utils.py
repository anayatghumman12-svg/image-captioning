import torch
from torchvision import transforms

import settings


def get_device():
    """Select and return the available PyTorch computation device.

    Returns:
        torch.device: Returns `cuda` device if GPU is available, otherwise `cpu`.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_transform():
    """Build the canonical image transformation pipeline for PyTorch models.

    This pipeline resizes images, converts them to PyTorch tensors, and applies
    ImageNet mean and standard deviation normalization. Used consistently during
    feature extraction and inference.

    Returns:
        transforms.Compose: PyTorch Torchvision composition of image transforms.
    """
    return transforms.Compose(
        [
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=settings.IMAGENET_MEAN, std=settings.IMAGENET_STD
            ),
        ]
    )