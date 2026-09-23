import torch
import torch.nn as nn
from torchvision import models

import settings


class ResNetEncoder(nn.Module):
    """ResNet-50 feature extractor for image captioning.

    Wraps a pre-trained ResNet-50 model, removes the final classification layer,
    and returns a 2048-dimensional global average pooled feature vector per image.
    """

    def __init__(self, freeze=True):
        """Initialize the ResNet-50 backbone and handle parameter freezing.

        Args:
            freeze (bool, optional): If True, freezes all backbone parameters and
                sets the network to evaluation mode. Defaults to True.
        """
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # resnet's layers in order end in (..., avgpool, fc) -- we drop the final fc (classifier)
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward(self, images):
        """Extract feature vectors from input image tensors.

        Args:
            images (torch.Tensor): Preprocessed image tensor batch of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Flattened feature embeddings of shape (batch_size, 2048).
        """
        with torch.no_grad():
            features = self.backbone(images)  # shape: (batch, 2048, 1, 1)
        return features.flatten(start_dim=1)  # shape: (batch, 2048)


if __name__ == "__main__":
    encoder = ResNetEncoder(freeze=True)
    dummy_image = torch.randn(
        1, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE
    )  # fake image to test shape
    output = encoder(dummy_image)
    print("Output shape:", output.shape)  # should be torch.Size([1, 2048])