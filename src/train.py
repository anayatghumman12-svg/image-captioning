import json
import os
import pickle
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import settings
from src.dataset import CaptionDataset, make_collate_fn
from src.decoder import LSTMDecoder
from src.preprocessing import load_captions, split_by_image
from src.utils import get_device
from src.vocabulary import Vocabulary


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Execute a single epoch of training or validation.

    Args:
        model (LSTMDecoder): The decoder neural network model.
        loader (DataLoader): PyTorch DataLoader delivering mini-batches.
        criterion (nn.Module): Loss function (CrossEntropyLoss).
        device (torch.device): Computation device (CPU or GPU).
        optimizer (torch.optim.Optimizer, optional): Optimizer for weight updates.
            If None, runs in evaluation mode without gradient updates. Defaults to None.

    Returns:
        float: Average loss value across all batches in the epoch.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.set_grad_enabled(is_training):
        for features, captions in loader:
            features, captions = features.to(device), captions.to(device)
            logits = model(features, captions)
            targets = captions[:, 1:]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings.GRAD_CLIP
                )
                optimizer.step()
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def train(train_captions, val_captions, train_features, val_features, vocab):
    """Train the LSTM decoder model over multiple epochs, saving best checkpoints and configs.

    Args:
        train_captions (dict): Dictionary of training image captions.
        val_captions (dict): Dictionary of validation image captions.
        train_features (dict): Pre-extracted CNN features for training set.
        val_features (dict): Pre-extracted CNN features for validation set.
        vocab (Vocabulary): Vocabulary mapping tokens to IDs.

    Returns:
        LSTMDecoder: The model instance loaded with best validation weights.
    """
    print("Starting train() function...")
    device = get_device()
    print("Training on device:", device)
    train_dataset = CaptionDataset(train_captions, train_features, vocab)
    val_dataset = CaptionDataset(val_captions, val_features, vocab)
    print(
        "Train examples:",
        len(train_dataset),
        "Val examples:",
        len(val_dataset),
    )
    collate_fn = make_collate_fn(vocab.pad_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = LSTMDecoder(
        vocab_size=len(vocab),
        embed_size=settings.EMBED_SIZE,
        hidden_size=settings.HIDDEN_SIZE,
        num_layers=settings.NUM_LSTM_LAYERS,
        pad_idx=vocab.pad_idx,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.LEARNING_RATE
    )
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)
    for epoch in range(1, settings.NUM_EPOCHS + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch}/{settings.NUM_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), settings.CHECKPOINT_PATH)
            print(f"  -> New best checkpoint saved (val_loss={val_loss:.4f})")
    config = {
        "vocab_size": len(vocab),
        "embed_size": settings.EMBED_SIZE,
        "hidden_size": settings.HIDDEN_SIZE,
        "feature_dim": settings.IMAGE_FEATURE_DIM,
        "num_layers": settings.NUM_LSTM_LAYERS,
        "pad_idx": vocab.pad_idx,
        "start_idx": vocab.start_idx,
        "end_idx": vocab.end_idx,
        "max_caption_length": settings.MAX_CAPTION_LENGTH,
    }
    with open(settings.CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(settings.LOSS_CURVE_PATH)
    print("Saved loss curve to", settings.LOSS_CURVE_PATH)
    model.load_state_dict(
        torch.load(settings.CHECKPOINT_PATH, map_location=device)
    )
    return model


if __name__ == "__main__":
    print("Script started")
    data = load_captions()
    train_captions, val_captions, test_captions = split_by_image(data)
    with open(os.path.join(settings.FEATURES_DIR, "train.pkl"), "rb") as f:
        train_features = pickle.load(f)
    with open(os.path.join(settings.FEATURES_DIR, "val.pkl"), "rb") as f:
        val_features = pickle.load(f)
    vocab = Vocabulary.load(settings.VOCAB_PATH)
    print("Loaded vocab, size:", len(vocab))
    train(train_captions, val_captions, train_features, val_features, vocab)
    print("Script finished")