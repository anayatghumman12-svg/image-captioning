import json
import re
from collections import Counter

import settings


def tokenize(caption):
    """Clean and split a caption string into a list of word tokens.

    Lowercases the input string, strips leading/trailing whitespace,
    removes non-alphanumeric characters, and splits on whitespace.

    Args:
        caption (str): Raw input caption string.

    Returns:
        list: List of cleaned word token strings.
    """
    caption = caption.lower().strip()
    caption = re.sub(r"[^a-z0-9\s]", "", caption)
    return caption.split()


class Vocabulary:
    """Manages mapping between word tokens and numerical IDs for image captioning.

    Handles special tokens (<pad>, <start>, <end>, <unk>), vocabulary building
    from corpus, numericalization, denumericalization, and persistence.
    """

    def __init__(self):
        """Initialize empty mapping dictionaries and register special tokens."""
        self.word2idx = {}
        self.idx2word = {}
        self._next_index = 0
        for special in (
            settings.PAD_TOKEN,
            settings.START_TOKEN,
            settings.END_TOKEN,
            settings.UNK_TOKEN,
        ):
            self._add_word(special)

    def _add_word(self, word):
        """Add a single word to the vocabulary mappings if not already present.

        Args:
            word (str): Word token string to register.
        """
        if word not in self.word2idx:
            self.word2idx[word] = self._next_index
            self.idx2word[self._next_index] = word
            self._next_index += 1

    def __len__(self):
        """Return the total size of the vocabulary.

        Returns:
            int: Number of unique tokens in vocabulary.
        """
        return len(self.word2idx)

    @property
    def pad_idx(self):
        """Return the numerical index of the padding token (<pad>)."""
        return self.word2idx[settings.PAD_TOKEN]

    @property
    def start_idx(self):
        """Return the numerical index of the start token (<start>)."""
        return self.word2idx[settings.START_TOKEN]

    @property
    def end_idx(self):
        """Return the numerical index of the end token (<end>)."""
        return self.word2idx[settings.END_TOKEN]

    @property
    def unk_idx(self):
        """Return the numerical index of the unknown token (<unk>)."""
        return self.word2idx[settings.UNK_TOKEN]

    def build(self, train_captions, min_freq=settings.MIN_WORD_FREQ):
        """Build vocabulary from training captions by filtering out low-frequency words.

        Args:
            train_captions (dict): Dictionary mapping image IDs to lists of caption strings.
            min_freq (int, optional): Minimum frequency threshold for a word to be included.
                Defaults to settings.MIN_WORD_FREQ.
        """
        counter = Counter()
        for captions in train_captions.values():
            for caption in captions:
                counter.update(tokenize(caption))
        for word, freq in counter.items():
            if freq >= min_freq:
                self._add_word(word)

    def numericalize(self, caption, max_length=settings.MAX_CAPTION_LENGTH):
        """Convert a raw caption string into a list of token IDs, bounded by <start> and <end>.

        Args:
            caption (str): Input text caption.
            max_length (int, optional): Maximum sequence length limit.
                Defaults to settings.MAX_CAPTION_LENGTH.

        Returns:
            list: Sequence of numerical token IDs.
        """
        tokens = tokenize(caption)
        ids = [self.start_idx]
        ids += [self.word2idx.get(tok, self.unk_idx) for tok in tokens]
        ids = ids[: max_length - 1]
        ids.append(self.end_idx)
        return ids

    def denumericalize(self, ids):
        """Convert a list of numerical token IDs back into a human-readable text string.

        Stops decoding upon encountering the <end> token and filters out <start> and <pad> tokens.

        Args:
            ids (list): Sequence of numerical token IDs.

        Returns:
            str: Decoded sentence string.
        """
        words = []
        for idx in ids:
            word = self.idx2word.get(idx, settings.UNK_TOKEN)
            if word == settings.END_TOKEN:
                break
            if word in (settings.START_TOKEN, settings.PAD_TOKEN):
                continue
            words.append(word)
        return " ".join(words)

    def save(self, path):
        """Save the word-to-index mapping dictionary to a JSON file.

        Args:
            path (str): File path where vocabulary JSON will be saved.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx}, f)

    @classmethod
    def load(cls, path):
        """Load and reconstruct a Vocabulary instance from a saved JSON file.

        Args:
            path (str): Path to the saved vocabulary JSON file.

        Returns:
            Vocabulary: Loaded vocabulary instance.
        """
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.word2idx = {k: int(v) for k, v in data["word2idx"].items()}
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        vocab._next_index = max(vocab.word2idx.values()) + 1
        return vocab


if __name__ == "__main__":
    import os
    from src.preprocessing import load_captions, split_by_image

    data = load_captions()
    train, val, test = split_by_image(data)
    vocab = Vocabulary()
    vocab.build(train)
    print("Vocabulary size:", len(vocab))

    # test numericalize -> denumericalize round trip
    sample_caption = "a dog is running in the grass"
    ids = vocab.numericalize(sample_caption)
    text_back = vocab.denumericalize(ids)
    print("Original:", sample_caption)
    print("As ids:  ", ids)
    print("Back to text:", text_back)

    # save it so we can use it later
    os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)
    vocab.save(settings.VOCAB_PATH)
    print("Saved vocab to", settings.VOCAB_PATH)