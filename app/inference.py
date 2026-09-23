import io
import json

import torch
from PIL import Image

import settings
from src.decoder import LSTMDecoder
from src.encoder import ResNetEncoder
from src.utils import get_device, get_image_transform
from src.vocabulary import Vocabulary


class CaptioningService:
    """Service class for loading trained models and generating image captions.

    Handles initialization of device, configurations, vocabulary, pre-trained
    CNN encoder, and trained LSTM decoder.
    """

    def __init__(self):
        """Initialize models, vocabulary, and configuration for inference."""
        self.device = get_device()
        self.transform = get_image_transform()

        with open(settings.CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.vocab = Vocabulary.load(settings.VOCAB_PATH)

        self.encoder = ResNetEncoder(freeze=True).to(self.device)
        self.encoder.eval()

        self.decoder = LSTMDecoder(
            vocab_size=self.config["vocab_size"],
            embed_size=self.config["embed_size"],
            hidden_size=self.config["hidden_size"],
            feature_dim=self.config["feature_dim"],
            num_layers=self.config["num_layers"],
            pad_idx=self.config["pad_idx"],
        ).to(self.device)
        self.decoder.load_state_dict(
            torch.load(settings.CHECKPOINT_PATH, map_location=self.device)
        )
        self.decoder.eval()

        print("CaptioningService loaded on device:", self.device)

    def caption_image(self, image_bytes: bytes) -> str:
        """Preprocess an uploaded image, run through encoder/decoder, and return caption.

        Args:
            image_bytes (bytes): Raw bytes of the uploaded image file.

        Returns:
            str: Generated textual caption for the given image.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.encoder(tensor)
            token_ids = self.decoder.generate_greedy(
                feature,
                start_idx=self.config["start_idx"],
                end_idx=self.config["end_idx"],
                max_length=self.config["max_caption_length"],
            )

        return self.vocab.denumericalize(token_ids)