"""Sanity checks and verification tests for CNN model architecture, saving, and loading."""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import build_character_cnn, load_trained_model, CLASS_MAPPING, NUM_CLASSES, INPUT_SHAPE
from src.dataset import load_emnist_letters
from src.preprocessing import preprocess_dataset_images

def test_model_architecture_and_sanity():
    print("=== TEST 1: Model Architecture & Sanity Checks ===")
    
    # 1. Build model
    model = build_character_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, include_augmentation=True)
    
    # 2. Check shapes
    assert model.input_shape == (None, 28, 28, 1), f"Unexpected input shape: {model.input_shape}"
    assert model.output_shape == (None, 26), f"Unexpected output shape: {model.output_shape}"
    print("  [PASS] Input shape (None, 28, 28, 1) and output shape (None, 26) verified.")

    # 3. Forward pass on dummy batch
    dummy_batch = np.random.uniform(0.0, 1.0, size=(4, 28, 28, 1)).astype(np.float32)
    preds = model(dummy_batch, training=False).numpy()
    
    assert preds.shape == (4, 26), f"Forward pass output shape error: {preds.shape}"
    assert 0.0 <= preds.min() and preds.max() <= 1.0, "Probabilities outside [0, 1]"
    # Verify softmax sum to 1.0
    sums = np.sum(preds, axis=-1)
    np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)
    print("  [PASS] Forward pass successful: Softmax probabilities sum to 1.0.")

    # 4. Check loss calculation
    dummy_labels = np.array([0, 5, 12, 25], dtype=np.int32)  # A, F, M, Z
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    loss_val = float(loss_fn(dummy_labels, preds).numpy())
    assert loss_val > 0, "Loss calculation failed"
    print(f"  [PASS] Loss calculation verified on labels [0, 25]: initial loss = {loss_val:.4f}.")

    # 5. Check authoritative class mapping
    assert len(CLASS_MAPPING) == 26
    assert CLASS_MAPPING[0] == "A" and CLASS_MAPPING[25] == "Z"
    print("  [PASS] Authoritative 26-class mapping (0->A ... 25->Z) verified.")

def test_saved_model_inference():
    print("\n=== TEST 2: Saved Model Load and Inference Test ===")
    model_path = os.path.join(PROJECT_ROOT, "models", "handwritten_character_model.keras")
    if not os.path.exists(model_path):
        print(f"  [SKIP] Model file {model_path} does not exist yet. Run training/train.py first.")
        return False

    # 1. Load model in clean context
    model = load_trained_model(model_path)
    print("  [PASS] Saved .keras model successfully loaded from disk.")

    # 2. Predict on 5 real unseen test images
    _, (X_test_raw, y_test_raw), _ = load_emnist_letters(fix_zero_indexed=True)
    sample_imgs = preprocess_dataset_images(X_test_raw[:5])
    sample_labels = y_test_raw[:5]

    probs = model.predict(sample_imgs, verbose=0)
    assert probs.shape == (5, 26), f"Prediction shape error: {probs.shape}"
    
    sums = np.sum(probs, axis=-1)
    np.testing.assert_allclose(sums, np.ones(5), atol=1e-5)
    
    top_indices = np.argmax(probs, axis=1)
    top_confidences = np.max(probs, axis=1)

    print("\n  Sample Predictions on Real Test Images:")
    for i in range(5):
        true_char = CLASS_MAPPING[sample_labels[i]]
        pred_char = CLASS_MAPPING[top_indices[i]]
        conf = top_confidences[i] * 100
        match = "CORRECT" if true_char == pred_char else "MISMATCH"
        print(f"    Sample {i+1}: True = '{true_char}', Pred = '{pred_char}' ({conf:.2f}%) [{match}]")

    print("  [PASS] Saved model inference test passed cleanly.")
    return True

if __name__ == "__main__":
    test_model_architecture_and_sanity()
    test_saved_model_inference()
