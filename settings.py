"""Global Configuration and Hyperparameters for Image Captioning Pipeline.

This module defines directory paths, dataset split configurations, vocabulary rules,
image preprocessing parameters, model architecture parameters, and training settings.
"""

import os

# Base Directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")

# Model & Artifact Output Paths
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "models")
VOCAB_PATH = os.path.join(ARTIFACTS_DIR, "vocab.json")
FEATURES_DIR = os.path.join(ARTIFACTS_DIR, "features")
CHECKPOINT_PATH = os.path.join(ARTIFACTS_DIR, "decoder_best.pt")
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
LOSS_CURVE_PATH = os.path.join(REPORTS_DIR, "loss_curve.png")

# Data Split Configuration
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SPLIT_SEED = 42

# Vocabulary & Text Processing Hyperparameters
MIN_WORD_FREQ = 5
MAX_CAPTION_LENGTH = 35
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

# Vision & Feature Extractor Settings
IMAGE_SIZE = 224
IMAGE_FEATURE_DIM = 2048
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Decoder Model Architecture
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LSTM_LAYERS = 1
DROPOUT = 0.5

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
GRAD_CLIP = 5.0