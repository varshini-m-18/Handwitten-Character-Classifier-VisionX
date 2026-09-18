"""Generate sample images for UI demo testing."""
import os
import sys
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import load_emnist_letters

SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "test_samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

def create_samples():
    print("Generating demo sample images in data/test_samples/...")
    
    # 1. Real EMNIST letters from test set
    _, (X_test, y_test), label_map = load_emnist_letters(fix_zero_indexed=True)
    
    # Save a few real EMNIST test characters
    for target_char in ["A", "B", "C", "M", "S", "Z"]:
        target_idx = ord(target_char) - ord("A")
        match_idx = np.where(y_test == target_idx)[0][0]
        emnist_img = Image.fromarray(X_test[match_idx], mode="L")
        # Save as 140x140 for nice preview
        emnist_large = emnist_img.resize((140, 140), Image.Resampling.NEAREST)
        path = os.path.join(SAMPLES_DIR, f"emnist_{target_char}.png")
        emnist_large.save(path)
        print(f"  Saved {path}")

    # Real ambiguous EMNIST test sample (sample 57, confidence 38.1%)
    amb_img = Image.fromarray(X_test[57], mode="L").resize((140, 140), Image.Resampling.NEAREST)
    amb_path = os.path.join(SAMPLES_DIR, "emnist_ambiguous_low_conf.png")
    amb_img.save(amb_path)
    print(f"  Saved {amb_path}")

    # 2. Paper-style handwritten drawings (White background, Black ink)
    for char in ["A", "H", "R"]:
        img = Image.new("RGB", (160, 160), color="white")
        draw = ImageDraw.Draw(img)
        w, h = 160, 160
        pad = int(w * 0.22)
        if char == "A":
            draw.line([(pad, h - pad), (w // 2, pad)], fill="black", width=10)
            draw.line([(w // 2, pad), (w - pad, h - pad)], fill="black", width=10)
            draw.line([(int(w * 0.35), int(h * 0.65)), (int(w * 0.65), int(h * 0.65))], fill="black", width=10)
        elif char == "H":
            draw.line([(pad, pad), (pad, h - pad)], fill="black", width=10)
            draw.line([(w - pad, pad), (w - pad, h - pad)], fill="black", width=10)
            draw.line([(pad, h // 2), (w - pad, h // 2)], fill="black", width=10)
        elif char == "R":
            draw.line([(pad, pad), (pad, h - pad)], fill="black", width=10)
            draw.arc([pad, pad, int(w * 0.8), int(h * 0.55)], start=270, end=90, fill="black", width=10)
            draw.line([(int(w * 0.5), int(h * 0.55)), (w - pad, h - pad)], fill="black", width=10)
        
        path = os.path.join(SAMPLES_DIR, f"paper_handwritten_{char}.png")
        img.save(path)
        print(f"  Saved {path}")

    # 3. Blank image for invalid defense demo
    blank_img = Image.new("RGB", (160, 160), color="white")
    blank_path = os.path.join(SAMPLES_DIR, "blank_sample.png")
    blank_img.save(blank_path)
    print(f"  Saved {blank_path}")

if __name__ == "__main__":
    import numpy as np
    create_samples()
