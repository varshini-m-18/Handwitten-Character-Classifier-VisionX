"""Tests for Stage 2 dataset verification."""
import os
import sys
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import load_emnist_letters, inspect_and_visualize_dataset, RESULTS_DIR

def test_stage2_dataset():
    print("Testing Stage 2 Dataset Pipeline...")
    stats = inspect_and_visualize_dataset()
    
    (X_train, y_train), (X_test, y_test), label_map = load_emnist_letters(fix_zero_indexed=True)
    
    # 1. Check dimensions
    assert len(X_train.shape) == 3, f"Expected 3D array for X_train, got {X_train.shape}"
    assert X_train.shape[1:] == (28, 28), f"Expected image shape (28, 28), got {X_train.shape[1:]}"
    assert len(X_test.shape) == 3, f"Expected 3D array for X_test, got {X_test.shape}"
    assert X_test.shape[1:] == (28, 28), f"Expected image shape (28, 28), got {X_test.shape[1:]}"
    
    # 2. Check counts
    assert len(X_train) == len(y_train), "Train images and labels count mismatch"
    assert len(X_test) == len(y_test), "Test images and labels count mismatch"
    
    # 3. Check classes
    assert len(label_map) == 26, f"Expected 26 classes, got {len(label_map)}"
    expected_letters = [chr(ord('A') + i) for i in range(26)]
    actual_letters = [label_map[i] for i in range(26)]
    assert actual_letters == expected_letters, f"Label map mismatch: {actual_letters} vs {expected_letters}"
    
    # 4. Check label range
    assert y_train.min() == 0 and y_train.max() == 25, f"Train labels out of range [0, 25]: min={y_train.min()}, max={y_train.max()}"
    assert y_test.min() == 0 and y_test.max() == 25, f"Test labels out of range [0, 25]: min={y_test.min()}, max={y_test.max()}"
    
    # 5. Check sample visualization exists
    sample_img_path = os.path.join(RESULTS_DIR, "dataset_samples.png")
    assert os.path.exists(sample_img_path), f"Sample visualization not found at {sample_img_path}"
    assert os.path.getsize(sample_img_path) > 1000, "Sample visualization file is empty"
    
    print("\n>>> ALL STAGE 2 DATASET TESTS PASSED SUCCESSFULLY! <<<")
    print(f"Verified {len(X_train):,} training samples and {len(X_test):,} testing samples across 26 classes (A-Z).")

if __name__ == "__main__":
    test_stage2_dataset()
