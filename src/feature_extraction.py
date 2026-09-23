import os
import pickle
import time
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import settings
from src.encoder import ResNetEncoder
from src.utils import get_device, get_image_transform


class _ImageOnlyDataset(Dataset):
    """Loads and preprocesses raw images for feature extraction.

    No captions are loaded or needed for this dataset class.
    """

    def __init__(self, filenames, images_dir):
        """Initialize dataset with image filenames, directory path, and transformation pipeline.

        Args:
            filenames (list): List of image filenames to process.
            images_dir (str): Path to the directory containing raw image files.
        """
        self.filenames = filenames
        self.images_dir = images_dir
        self.transform = get_image_transform()

    def __len__(self):
        """Return total number of images in the dataset.

        Returns:
            int: Number of image filenames.
        """
        return len(self.filenames)

    def __getitem__(self, index):
        """Retrieve preprocessed image tensor and its filename by index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: (transformed_image_tensor, filename_string).
        """
        filename = self.filenames[index]
        image_path = os.path.join(self.images_dir, filename)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), filename


def extract_and_cache_features(
    filenames,
    images_dir=settings.IMAGES_DIR,
    output_path=None,
    batch_size=64,
):
    """Extract 2048-dimensional CNN features for a list of images and optionally cache them to disk.

    Args:
        filenames (list): List of image filenames to extract features from.
        images_dir (str, optional): Root directory containing raw images. Defaults to settings.IMAGES_DIR.
        output_path (str, optional): File path to save pickled features dictionary. Defaults to None.
        batch_size (int, optional): Batch size for PyTorch DataLoader. Defaults to 64.

    Returns:
        dict: Mapping of image filenames to extracted 1D PyTorch feature tensors.
    """
    device = get_device()
    encoder = ResNetEncoder(freeze=True).to(device)
    encoder.eval()

    dataset = _ImageOnlyDataset(filenames, images_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features = {}
    start_time = time.time()
    for images, batch_filenames in tqdm(loader, desc="Extracting features"):
        images = images.to(device)
        batch_features = encoder(images).cpu()
        for filename, feature in zip(batch_filenames, batch_features):
            features[filename] = feature
    elapsed = time.time() - start_time
    print(f"Extracted {len(features)} features in {elapsed:.1f}s")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(features, f)
        print("Saved features to", output_path)

    return features


if __name__ == "__main__":
    from src.preprocessing import load_captions, split_by_image

    data = load_captions()
    train, val, test = split_by_image(data)

    for split_name, captions_dict in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        filenames = list(captions_dict.keys())
        output_path = os.path.join(
            settings.ARTIFACTS_DIR, "features", f"{split_name}.pkl"
        )
        print(
            f"\n--- Extracting features for {split_name} split ({len(filenames)} images) ---"
        )
        extract_and_cache_features(filenames, output_path=output_path)