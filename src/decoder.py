import pickle
import torch
import torch.nn as nn

import settings
from src.vocabulary import Vocabulary


class LSTMDecoder(nn.Module):
    """LSTM-based decoder for generating text captions conditioned on image features."""

    def __init__(
        self,
        vocab_size,
        embed_size=256,
        hidden_size=512,
        feature_dim=settings.IMAGE_FEATURE_DIM,
        num_layers=1,
        pad_idx=0,
    ):
        """Initialize embedding layer, linear projections, LSTM cell, and output classifier.

        Args:
            vocab_size (int): Total size of the target vocabulary.
            embed_size (int, optional): Embedding dimension for input words. Defaults to 256.
            hidden_size (int, optional): Number of features in the LSTM hidden state. Defaults to 512.
            feature_dim (int, optional): Dimension of input CNN feature vectors. Defaults to settings.IMAGE_FEATURE_DIM.
            num_layers (int, optional): Number of recurrent layers in LSTM. Defaults to 1.
            pad_idx (int, optional): Index of the padding token (<pad>). Defaults to 0.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocab_size, embed_size, padding_idx=pad_idx
        )
        self.feature_to_h0 = nn.Linear(feature_dim, hidden_size * num_layers)
        self.feature_to_c0 = nn.Linear(feature_dim, hidden_size * num_layers)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def _init_hidden(self, features):
        """Project image feature vector into starting hidden and cell states (h0, c0) for the LSTM.

        Args:
            features (torch.Tensor): Image feature tensor of shape (batch_size, feature_dim).

        Returns:
            tuple: Initial (h0, c0) state tensors formatted for PyTorch's LSTM.
        """
        batch_size = features.size(0)
        h0 = torch.tanh(self.feature_to_h0(features))
        c0 = torch.tanh(self.feature_to_c0(features))
        h0 = (
            h0.view(batch_size, self.num_layers, self.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        c0 = (
            c0.view(batch_size, self.num_layers, self.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        return h0, c0

    def forward(self, features, captions):
        """Perform training-time forward pass using teacher forcing.

        Args:
            features (torch.Tensor): Image features of shape (batch_size, feature_dim).
            captions (torch.Tensor): Ground-truth caption token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Raw output logits for predicting captions[:, 1:] of shape (batch_size, seq_len - 1, vocab_size).
        """
        h0, c0 = self._init_hidden(features)
        decoder_input = captions[:, :-1]  # everything except the LAST token
        embeddings = self.embedding(decoder_input)
        outputs, _ = self.lstm(embeddings, (h0, c0))
        logits = self.fc_out(outputs)
        return (
            logits  # predicts captions[:, 1:] i.e. everything except the FIRST token
        )

    @torch.no_grad()
    def generate_greedy(
        self, features, start_idx, end_idx, max_length=settings.MAX_CAPTION_LENGTH
    ):
        """Perform greedy autoregressive decoding at inference time.

        Args:
            features (torch.Tensor): Single image feature vector of shape (1, feature_dim).
            start_idx (int): Token ID representing the start of a caption (<start>).
            end_idx (int): Token ID representing the end of a caption (<end>).
            max_length (int, optional): Maximum sequence length to generate. Defaults to settings.MAX_CAPTION_LENGTH.

        Returns:
            list: Generated token ID sequence including start and end markers.
        """
        device = features.device
        h, c = self._init_hidden(features)
        input_token = torch.tensor([[start_idx]], device=device)
        generated = [start_idx]

        for _ in range(max_length - 1):
            embedded = self.embedding(input_token)
            output, (h, c) = self.lstm(embedded, (h, c))
            logits = self.fc_out(output.squeeze(1))
            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)
            if next_token == end_idx:
                break
            input_token = torch.tensor([[next_token]], device=device)

        return generated


if __name__ == "__main__":
    vocab = Vocabulary.load(settings.VOCAB_PATH)

    with open(settings.ARTIFACTS_DIR + "/features/train.pkl", "rb") as f:
        train_features = pickle.load(f)

    model = LSTMDecoder(vocab_size=len(vocab), pad_idx=vocab.pad_idx)

    # fake a small batch: 2 images, captions already padded
    fake_features = torch.stack(list(train_features.values())[:2])
    fake_captions = torch.tensor(
        [
            [vocab.start_idx, 5, 6, 7, vocab.end_idx, vocab.pad_idx],
            [
                vocab.start_idx,
                8,
                9,
                vocab.end_idx,
                vocab.pad_idx,
                vocab.pad_idx,
            ],
        ]
    )

    logits = model(fake_features, fake_captions)
    print("Logits shape:", logits.shape)  # expect (2, 5, vocab_size)

    # test greedy generation on ONE image
    single_feature = fake_features[0:1]
    generated_ids = model.generate_greedy(
        single_feature, vocab.start_idx, vocab.end_idx
    )
    print("Generated ids:", generated_ids)
    print("As text:", vocab.denumericalize(generated_ids))