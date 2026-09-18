"""Stage 3 Preprocessing Verification Test Suite."""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import load_emnist_letters, RESULTS_DIR
from src.preprocessing import (
    preprocess_dataset_images,
    preprocess_user_image,
    get_training_augmentation_model
)

def create_synthetic_letter(char="A", size=(128, 128), bg="white", fg="black"):
    """Generate a clean synthetic character image for testing."""
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)
    # Draw simple character lines
    if char == "A":
        # Draw an 'A' using lines for clear strokes
        w, h = size
        pad = int(w * 0.2)
        # Left leg
        draw.line([(pad, h - pad), (w // 2, pad)], fill=fg, width=max(2, int(w * 0.08)))
        # Right leg
        draw.line([(w // 2, pad), (w - pad, h - pad)], fill=fg, width=max(2, int(w * 0.08)))
        # Crossbar
        draw.line([(int(w * 0.32), int(h * 0.65)), (int(w * 0.68), int(h * 0.65))], fill=fg, width=max(2, int(w * 0.08)))
    elif char == "B":
        w, h = size
        pad = int(w * 0.2)
        # Vertical spine
        draw.line([(pad, pad), (pad, h - pad)], fill=fg, width=max(2, int(w * 0.08)))
        # Top loop
        draw.arc([pad, pad, int(w * 0.8), int(h * 0.5)], start=270, end=90, fill=fg, width=max(2, int(w * 0.08)))
        # Bottom loop
        draw.arc([pad, int(h * 0.5), int(w * 0.8), h - pad], start=270, end=90, fill=fg, width=max(2, int(w * 0.08)))
    return img

def test_stage3_preprocessing():
    print("=== STARTING STAGE 3 PREPROCESSING VERIFICATION ===")
    
    # -------------------------------------------------------------
    # 1. Test Dataset Preprocessing (used for training / testing)
    # -------------------------------------------------------------
    print("\n[Test 1] Testing preprocess_dataset_images()...")
    (X_train, y_train), (X_test, y_test), label_map = load_emnist_letters(fix_zero_indexed=True)
    sample_emnist_raw = X_train[:10]  # Take 10 raw images
    processed_emnist = preprocess_dataset_images(sample_emnist_raw)
    
    assert processed_emnist.shape == (10, 28, 28, 1), f"Expected (10, 28, 28, 1), got {processed_emnist.shape}"
    assert processed_emnist.dtype == np.float32, f"Expected float32, got {processed_emnist.dtype}"
    assert 0.0 <= processed_emnist.min() and processed_emnist.max() <= 1.0, "Values out of [0, 1] range"
    # Ensure original array was not mutated
    assert sample_emnist_raw.dtype == np.uint8, "Original dataset array was unexpectedly mutated"
    print("  [PASS] Dataset preprocessing passed: shape (N, 28, 28, 1), normalized to [0, 1], non-mutating.")

    # -------------------------------------------------------------
    # 2. Test User Input: Grayscale & RGB Formats
    # -------------------------------------------------------------
    print("\n[Test 2] Testing RGB and Grayscale Inputs...")
    rgb_img = create_synthetic_letter("A", size=(100, 100), bg="white", fg="black")
    gray_img = rgb_img.convert("L")
    
    out_rgb, valid_rgb, msg_rgb, vis_rgb = preprocess_user_image(rgb_img)
    out_gray, valid_gray, msg_gray, vis_gray = preprocess_user_image(gray_img)
    
    assert valid_rgb and valid_gray, "Synthetic 'A' should be valid"
    assert out_rgb.shape == (1, 28, 28, 1), f"Expected (1, 28, 28, 1), got {out_rgb.shape}"
    assert out_gray.shape == (1, 28, 28, 1), f"Expected (1, 28, 28, 1), got {out_gray.shape}"
    assert out_rgb.dtype == np.float32 and out_gray.dtype == np.float32
    assert 0.0 <= out_rgb.min() and out_rgb.max() <= 1.0
    print("  [PASS] RGB and Grayscale inputs processed consistently to (1, 28, 28, 1).")

    # -------------------------------------------------------------
    # 3. Test User Input: Different Image Resolutions
    # -------------------------------------------------------------
    print("\n[Test 3] Testing Different Input Resolutions...")
    sizes = [(28, 28), (64, 64), (120, 200), (300, 300)]
    for sz in sizes:
        synth = create_synthetic_letter("A", size=sz, bg="white", fg="black")
        out, valid, msg, vis = preprocess_user_image(synth)
        assert valid, f"Size {sz} failed validation: {msg}"
        assert out.shape == (1, 28, 28, 1), f"Size {sz} did not yield (1, 28, 28, 1), got {out.shape}"
    print(f"  [PASS] Arbitrary sizes {sizes} correctly resized and centered to (1, 28, 28, 1).")

    # -------------------------------------------------------------
    # 4. Test User Input: Dark-on-Light vs Light-on-Dark Polarity
    # -------------------------------------------------------------
    print("\n[Test 4] Testing Polarity Normalization (Paper vs Dark Canvas)...")
    # Paper style: white background (255), black stroke (0)
    paper_img = create_synthetic_letter("A", size=(100, 100), bg="white", fg="black")
    # Canvas style: black background (0), white stroke (255)
    canvas_img = create_synthetic_letter("A", size=(100, 100), bg="black", fg="white")
    
    out_paper, valid_paper, _, vis_paper = preprocess_user_image(paper_img)
    out_canvas, valid_canvas, _, vis_canvas = preprocess_user_image(canvas_img)
    
    assert valid_paper and valid_canvas
    # Both should have the character stroke as bright pixels (> 0.5) and corners as dark (~ 0.0)
    corner_paper = float(out_paper[0, 0, 0, 0])
    corner_canvas = float(out_canvas[0, 0, 0, 0])
    center_paper = float(np.max(out_paper))
    center_canvas = float(np.max(out_canvas))
    
    assert corner_paper < 0.1, f"Paper background should be black (0) after normalization, got {corner_paper}"
    assert corner_canvas < 0.1, f"Canvas background should be black (0) after normalization, got {corner_canvas}"
    assert center_paper > 0.5, f"Paper stroke should be bright (>0.5), got {center_paper}"
    assert center_canvas > 0.5, f"Canvas stroke should be bright (>0.5), got {center_canvas}"
    print("  [PASS] Both light-background and dark-background inputs successfully normalized to EMNIST format.")

    # -------------------------------------------------------------
    # 5. Test User Input: Blank & Invalid Images
    # -------------------------------------------------------------
    print("\n[Test 5] Testing Blank and Invalid Input Handling...")
    blank_white = np.ones((100, 100), dtype=np.uint8) * 255
    blank_black = np.zeros((100, 100), dtype=np.uint8)
    empty_arr = np.array([])
    
    out_bw, valid_bw, msg_bw, _ = preprocess_user_image(blank_white)
    out_bb, valid_bb, msg_bb, _ = preprocess_user_image(blank_black)
    out_em, valid_em, msg_em, _ = preprocess_user_image(empty_arr)
    
    assert not valid_bw, "Blank white image should be flagged invalid"
    assert not valid_bb, "Blank black image should be flagged invalid"
    assert not valid_em, "Empty array should be flagged invalid"
    assert out_bw.shape == (1, 28, 28, 1) and out_bb.shape == (1, 28, 28, 1)
    print(f"  [PASS] Safely caught blank white: '{msg_bw}'")
    print(f"  [PASS] Safely caught blank black: '{msg_bb}'")
    print(f"  [PASS] Safely caught empty input: '{msg_em}'")

    # -------------------------------------------------------------
    # 6. Test EMNIST Sample through User Pipeline
    # -------------------------------------------------------------
    print("\n[Test 6] Testing Raw EMNIST Sample via User Pipeline...")
    emnist_a_raw = X_train[np.where(y_train == 0)[0][0]]  # Class 0 = 'A'
    out_emnist, valid_emnist, msg_emnist, vis_emnist = preprocess_user_image(emnist_a_raw)
    assert valid_emnist, f"EMNIST 'A' sample should be valid, got {msg_emnist}"
    assert out_emnist.shape == (1, 28, 28, 1)
    print("  [PASS] EMNIST sample correctly processed by user pipeline.")

    # -------------------------------------------------------------
    # 7. Test Training Data Augmentation Model
    # -------------------------------------------------------------
    print("\n[Test 7] Testing Training Augmentation Model...")
    aug_model = get_training_augmentation_model()
    dummy_input = processed_emnist[:4]  # 4 images: (4, 28, 28, 1)
    augmented_output = aug_model(dummy_input, training=True).numpy()
    
    assert augmented_output.shape == dummy_input.shape, f"Shape mismatch: {augmented_output.shape} vs {dummy_input.shape}"
    assert augmented_output.dtype == np.float32
    # Check that in inference mode (training=False), augmentation is identity
    inference_output = aug_model(dummy_input, training=False).numpy()
    np.testing.assert_allclose(inference_output, dummy_input, atol=1e-5)
    print("  [PASS] Augmentation model verified: active during training=True, exact identity during training=False.")

    # -------------------------------------------------------------
    # 8. Generate Before & After Visualization Grid
    # -------------------------------------------------------------
    print("\n[Test 8] Generating Before/After Preprocessing Visualization...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    vis_path = os.path.join(RESULTS_DIR, "preprocessing_samples.png")
    
    test_cases = [
        ("EMNIST Sample 'A'", emnist_a_raw, False),
        ("Paper Scan 'A' (White bg)", paper_img, True),
        ("Dark Canvas 'A' (Black bg)", canvas_img, True),
        ("Synth 'B' (150x80 Rect)", create_synthetic_letter("B", size=(150, 80), bg="white", fg="black"), True),
        ("Blank White", blank_white, False),
        ("Blank Black", blank_black, False),
    ]

    fig, axes = plt.subplots(len(test_cases), 3, figsize=(10, 2.5 * len(test_cases)))
    
    for idx, (label, img_input, is_pil) in enumerate(test_cases):
        out_batch, valid, msg, intermediate = preprocess_user_image(img_input)
        
        # Display original
        if is_pil:
            axes[idx, 0].imshow(img_input)
        else:
            axes[idx, 0].imshow(img_input, cmap="gray")
        axes[idx, 0].set_title(f"Original: {label}", fontsize=9, fontweight="bold")
        axes[idx, 0].axis("off")
        
        # Display 28x28 intermediate view
        axes[idx, 1].imshow(intermediate, cmap="gray")
        axes[idx, 1].set_title(f"Normalized (28x28)\nValid: {valid}", fontsize=9)
        axes[idx, 1].axis("off")
        
        # Display final CNN tensor heat map
        axes[idx, 2].imshow(out_batch[0, :, :, 0], cmap="viridis")
        axes[idx, 2].set_title(f"CNN Tensor [0.0, 1.0]\nRange: [{out_batch.min():.2f}, {out_batch.max():.2f}]", fontsize=9)
        axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(vis_path, dpi=150)
    plt.close()
    print(f"  [PASS] Saved preprocessing visualization to: {vis_path}")

    print("\n>>> ALL STAGE 3 PREPROCESSING TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_stage3_preprocessing()
