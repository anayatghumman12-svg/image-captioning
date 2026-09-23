import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """Pairs cached CNN features with numericalized captions.

    Each (image, one caption) pair is treated as a separate training example.
    """

    def __init__(self, captions_dict, features, vocab):
        """Initialize the dataset with captions, cached features, and vocabulary.

        Args:
            captions_dict (dict): Dictionary mapping image filenames to lists of captions.
            features (dict): Dictionary mapping image filenames to cached CNN feature tensors.
            vocab (Vocabulary): Vocabulary instance used to numericalize caption strings.
        """
        self.vocab = vocab
        self.examples = []
        for filename, captions in captions_dict.items():
            if filename not in features:
                continue
            for caption in captions:
                self.examples.append((filename, caption))
        self.features = features

    def __len__(self):
        """Return the total number of training examples.

        Returns:
            int: Total count of (image, caption) pairs in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, index):
        """Retrieve the CNN feature tensor and numericalized caption for a given index.

        Args:
            index (int): Index of the example to retrieve.

        Returns:
            tuple: (feature_tensor, caption_tensor) where caption_tensor contains token IDs.
        """
        filename, caption = self.examples[index]
        feature = self.features[filename]
        token_ids = self.vocab.numericalize(caption)
        return feature, torch.tensor(token_ids, dtype=torch.long)


def make_collate_fn(pad_idx):
    """Build a collate function that pads variable-length captions in a batch.

    Args:
        pad_idx (int): The index of the padding token (<pad>) in the vocabulary.

    Returns:
        function: A custom collate_fn for PyTorch DataLoader to handle dynamic padding.
    """

    def collate_fn(batch):
        """Collate individual samples into a padded batch.

        Args:
            batch (list): List of tuples containing (feature_tensor, caption_tensor).

        Returns:
            tuple: (stacked_features, padded_captions) tensors for mini-batch processing.
        """
        features, captions = zip(*batch)
        features = torch.stack(features, dim=0)
        captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
        return features, captions

    return collate_fn


if __name__ == "__main__":
    import pickle
    import settings
    from src.preprocessing import load_captions, split_by_image
    from src.vocabulary import Vocabulary

    data = load_captions()
    train, val, test = split_by_image(data)

    with open(settings.ARTIFACTS_DIR + "/features/train.pkl", "rb") as f:
        train_features = pickle.load(f)

    vocab = Vocabulary.load(settings.VOCAB_PATH)

    dataset = CaptionDataset(train, train_features, vocab)
    print("Total training examples:", len(dataset))

    feature, caption_ids = dataset[0]
    print("Feature shape:", feature.shape)
    print("Caption ids:", caption_ids)

    from torch.utils.data import DataLoader

    collate_fn = make_collate_fn(vocab.pad_idx)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch_features, batch_captions = next(iter(loader))
    print("Batch features shape:", batch_features.shape)
    print("Batch captions shape:", batch_captions.shape)