import csv
import random
import settings


def load_captions():
    """Read the dataset CSV file and group captions by image filename.

    Returns:
        dict: Mapping of image filenames to lists of ground-truth caption strings.
    """
    grouped = {}

    with open(settings.CAPTIONS_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip "image,caption" header line

        for row in reader:
            image_name = row[0].strip()
            caption = row[1].strip()

            if image_name not in grouped:
                grouped[image_name] = []
            grouped[image_name].append(caption)

    return grouped


def split_by_image(
    grouped_captions,
    train_ratio=settings.TRAIN_RATIO,
    val_ratio=settings.VAL_RATIO,
    seed=settings.SPLIT_SEED,
):
    """Split images into train, validation, and test sets to prevent data leakage across captions.

    Args:
        grouped_captions (dict): Dictionary mapping image filenames to caption lists.
        train_ratio (float, optional): Proportion of data for training. Defaults to settings.TRAIN_RATIO.
        val_ratio (float, optional): Proportion of data for validation. Defaults to settings.VAL_RATIO.
        seed (int, optional): Random seed for reproducible shuffling. Defaults to settings.SPLIT_SEED.

    Returns:
        tuple: Three dictionaries (train, val, test) mapping image filenames to their respective captions.
    """
    filenames = sorted(
        grouped_captions.keys()
    )  # sort first so shuffle is reproducible
    rng = random.Random(seed)
    rng.shuffle(filenames)

    n_total = len(filenames)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = filenames[:n_train]
    val_files = filenames[n_train : n_train + n_val]
    test_files = filenames[n_train + n_val :]

    train = {f: grouped_captions[f] for f in train_files}
    val = {f: grouped_captions[f] for f in val_files}
    test = {f: grouped_captions[f] for f in test_files}

    return train, val, test


if __name__ == "__main__":
    data = load_captions()
    print("Total images:", len(data))

    train, val, test = split_by_image(data)
    print("Train:", len(train), "Val:", len(val), "Test:", len(test))