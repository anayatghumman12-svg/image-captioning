import json
import os
import pickle
import time

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

import settings
from src.decoder import LSTMDecoder
from src.preprocessing import load_captions, split_by_image
from src.utils import get_device
from src.vocabulary import Vocabulary, tokenize


def generate_all_captions(model, features, vocab, device):
    """Generate greedy captions for all image features in a dataset split.

    Args:
        model (LSTMDecoder): Trained decoder model for inference.
        features (dict): Dictionary mapping image filenames to CNN feature tensors.
        vocab (Vocabulary): Vocabulary instance for decoding token IDs to text.
        device (torch.device): Computation device (CPU or GPU).

    Returns:
        dict: Dictionary mapping image filenames to generated text captions.
    """
    model.eval()
    generated = {}
    for filename, feature in features.items():
        feature = feature.unsqueeze(0).to(device)
        token_ids = model.generate_greedy(
            feature, vocab.start_idx, vocab.end_idx
        )
        generated[filename] = vocab.denumericalize(token_ids)
    return generated


def compute_bleu_scores(generated, references):
    """Compute corpus-level BLEU-1 through BLEU-4 evaluation metrics.

    Args:
        generated (dict): Dictionary mapping image filenames to generated text captions.
        references (dict): Dictionary mapping image filenames to lists of ground-truth reference captions.

    Returns:
        dict: Calculated corpus BLEU scores from BLEU-1 to BLEU-4.
    """
    hypotheses, list_of_references = [], []
    for filename, caption in generated.items():
        if filename not in references:
            continue
        hypotheses.append(tokenize(caption))
        list_of_references.append(
            [tokenize(ref) for ref in references[filename]]
        )

    smoothing = SmoothingFunction().method1
    scores = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n for _ in range(n)) + tuple(
            0.0 for _ in range(4 - n)
        )
        scores[f"bleu{n}"] = corpus_bleu(
            list_of_references,
            hypotheses,
            weights=weights,
            smoothing_function=smoothing,
        )
    return scores


if __name__ == "__main__":
    print("Evaluation started")

    data = load_captions()
    _, _, test_captions = split_by_image(data)

    with open(os.path.join(settings.FEATURES_DIR, "test.pkl"), "rb") as f:
        test_features = pickle.load(f)

    vocab = Vocabulary.load(settings.VOCAB_PATH)

    with open(settings.CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = get_device()
    model = LSTMDecoder(
        vocab_size=config["vocab_size"],
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        feature_dim=config["feature_dim"],
        num_layers=config["num_layers"],
        pad_idx=config["pad_idx"],
    ).to(device)
    model.load_state_dict(
        torch.load(settings.CHECKPOINT_PATH, map_location=device)
    )
    print("Model loaded successfully")

    start = time.time()
    generated = generate_all_captions(model, test_features, vocab, device)
    print(
        f"Generated {len(generated)} captions in {time.time() - start:.1f}s"
    )

    scores = compute_bleu_scores(generated, test_captions)
    print("BLEU scores:", scores)

    # save a report for later (README + analysis answers)
    report = {"bleu_scores": scores, "examples": []}
    print("\n--- 5 example captions ---")
    for filename in list(test_captions.keys())[:5]:
        example = {
            "image": filename,
            "references": test_captions[filename],
            "generated": generated.get(filename, "N/A"),
        }
        report["examples"].append(example)
        print(f"\nImage: {filename}")
        print("References:", test_captions[filename][:2])
        print("Generated: ", generated.get(filename, "N/A"))

    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    with open(
        os.path.join(settings.REPORTS_DIR, "evaluation_report.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)
    print("\nSaved report to reports/evaluation_report.json")