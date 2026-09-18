"""Dataset loading, verification, and inspection for EMNIST Letters (A-Z)."""
import os
import gzip
import struct
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Official EMNIST Letters mirror on Hugging Face (Royc30ne/emnist-letters)
BASE_URL = "https://huggingface.co/datasets/Royc30ne/emnist-letters/resolve/main/"
FILES = {
    "train_images": "emnist-letters-train-images-idx3-ubyte.gz",
    "train_labels": "emnist-letters-train-labels-idx1-ubyte.gz",
    "test_images": "emnist-letters-test-images-idx3-ubyte.gz",
    "test_labels": "emnist-letters-test-labels-idx1-ubyte.gz",
    "mapping": "emnist-letters-mapping.txt",
}

def download_file(url: str, dest_path: str):
    """Download a file if it does not already exist."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"File already exists: {os.path.basename(dest_path)}")
        return

    print(f"Downloading {os.path.basename(dest_path)} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
    print(f"Downloaded {os.path.basename(dest_path)} successfully ({os.path.getsize(dest_path)} bytes).")

def download_dataset():
    """Download all EMNIST Letters dataset files."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    for key, filename in FILES.items():
        url = BASE_URL + filename
        dest = os.path.join(RAW_DATA_DIR, filename)
        download_file(url, dest)

def read_idx_images(filepath: str) -> np.ndarray:
    """Read IDX3 ubyte image file from gzip archive and fix EMNIST 90-deg transposition."""
    with gzip.open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {filepath}")
        buf = f.read(num_images * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        # EMNIST binary files store images rotated 90 deg clockwise and horizontal flipped.
        # Transposing axes 1 and 2 corrects the orientation so characters are upright.
        data = np.transpose(data, (0, 2, 1))
        return data

def read_idx_labels(filepath: str) -> np.ndarray:
    """Read IDX1 ubyte label file from gzip archive."""
    with gzip.open(filepath, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {filepath}")
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

def read_mapping(filepath: str) -> dict:
    """Read EMNIST letters mapping file and verify class labels."""
    mapping = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                class_id = int(parts[0])
                ascii_code = int(parts[1])
                char = chr(ascii_code)
                mapping[class_id] = char
    return mapping

def load_emnist_letters(fix_zero_indexed: bool = True):
    """
    Load EMNIST Letters train and test sets.
    
    In raw EMNIST Letters:
      - Labels are 1 to 26 representing 'A' through 'Z'.
    
    If fix_zero_indexed is True:
      - Labels are converted to 0 to 25 where 0='A', 1='B', ..., 25='Z'.
    
    Returns:
      (X_train, y_train), (X_test, y_test), label_map
    """
    download_dataset()

    train_img_path = os.path.join(RAW_DATA_DIR, FILES["train_images"])
    train_lbl_path = os.path.join(RAW_DATA_DIR, FILES["train_labels"])
    test_img_path = os.path.join(RAW_DATA_DIR, FILES["test_images"])
    test_lbl_path = os.path.join(RAW_DATA_DIR, FILES["test_labels"])
    mapping_path = os.path.join(RAW_DATA_DIR, FILES["mapping"])

    X_train = read_idx_images(train_img_path)
    y_train = read_idx_labels(train_lbl_path)
    X_test = read_idx_images(test_img_path)
    y_test = read_idx_labels(test_lbl_path)
    raw_mapping = read_mapping(mapping_path)

    if fix_zero_indexed:
        # Original EMNIST letters: 1..26 -> 0..25
        y_train = y_train - 1
        y_test = y_test - 1
        label_map = {cid - 1: char for cid, char in raw_mapping.items() if 1 <= cid <= 26}
    else:
        label_map = raw_mapping

    return (X_train, y_train), (X_test, y_test), label_map

def inspect_and_visualize_dataset():
    """Verify dataset statistics and generate sample visualization grid for A-Z."""
    print("--- Loading EMNIST Letters Dataset ---")
    (X_train, y_train), (X_test, y_test), label_map = load_emnist_letters(fix_zero_indexed=True)

    print("\n--- DATASET VERIFICATION RESULTS ---")
    print(f"X_train shape: {X_train.shape} (dtype: {X_train.dtype})")
    print(f"y_train shape: {y_train.shape} (dtype: {y_train.dtype})")
    print(f"X_test shape:  {X_test.shape} (dtype: {X_test.dtype})")
    print(f"y_test shape:  {y_test.shape} (dtype: {y_test.dtype})")
    print(f"Number of classes: {len(label_map)}")
    print(f"Label min: {y_train.min()}, max: {y_train.max()}")
    print(f"Pixel min: {X_train.min()}, max: {X_train.max()}")

    # Verify each class 0..25
    print("\nClass mapping (0 to 25 -> A to Z):")
    mapping_summary = [f"{idx}:{label_map[idx]}" for idx in range(26)]
    print(", ".join(mapping_summary[:13]))
    print(", ".join(mapping_summary[13:]))

    # Verify class balance
    train_counts = np.bincount(y_train, minlength=26)
    test_counts = np.bincount(y_test, minlength=26)
    print(f"\nTrain samples per class: min={train_counts.min()}, max={train_counts.max()} (balanced: {train_counts.min() == train_counts.max()})")
    print(f"Test samples per class:  min={test_counts.min()}, max={test_counts.max()} (balanced: {test_counts.min() == test_counts.max()})")

    # Generate sample visualization for each class A-Z
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sample_img_path = os.path.join(RESULTS_DIR, "dataset_samples.png")
    
    fig, axes = plt.subplots(4, 7, figsize=(14, 8))
    axes = axes.flatten()

    for idx in range(26):
        # Pick the first sample of class idx
        sample_indices = np.where(y_train == idx)[0]
        sample_img = X_train[sample_indices[0]]
        char_label = label_map[idx]
        
        axes[idx].imshow(sample_img, cmap="gray")
        axes[idx].set_title(f"Class {idx}: '{char_label}'", fontsize=10, fontweight="bold")
        axes[idx].axis("off")

    # Hide unused subplots (26 classes in 28 subplots)
    for idx in range(26, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("EMNIST Letters Sample Visualizations (Classes 0-25: A-Z)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(sample_img_path, dpi=150)
    plt.close()
    print(f"\nSaved sample visualizations to: {sample_img_path}")

    return {
        "num_classes": len(label_map),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "image_shape": X_train.shape[1:],
        "train_per_class": int(train_counts[0]),
        "test_per_class": int(test_counts[0]),
    }

if __name__ == "__main__":
    inspect_and_visualize_dataset()
