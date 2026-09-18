"""Unit tests for Stage 5 inference pipeline, Top-3, confidence thresholding, and input defense."""
import os
import io
import sys
import numpy as np
from PIL import Image, ImageDraw

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import load_emnist_letters
from src.prediction import CharacterPredictor, predict_character
from src.model import CLASS_MAPPING

def draw_test_character(char="A", size=(120, 120), bg="white", fg="black") -> Image.Image:
    """Helper to draw clean test characters."""
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)
    w, h = size
    pad = int(w * 0.2)
    if char == "A":
        draw.line([(pad, h - pad), (w // 2, pad)], fill=fg, width=max(3, int(w * 0.08)))
        draw.line([(w // 2, pad), (w - pad, h - pad)], fill=fg, width=max(3, int(w * 0.08)))
        draw.line([(int(w * 0.32), int(h * 0.65)), (int(w * 0.68), int(h * 0.65))], fill=fg, width=max(3, int(w * 0.08)))
    elif char == "B":
        draw.line([(pad, pad), (pad, h - pad)], fill=fg, width=max(3, int(w * 0.08)))
        draw.arc([pad, pad, int(w * 0.8), int(h * 0.5)], start=270, end=90, fill=fg, width=max(3, int(w * 0.08)))
        draw.arc([pad, int(h * 0.5), int(w * 0.8), h - pad], start=270, end=90, fill=fg, width=max(3, int(w * 0.08)))
    return img

def test_stage5_inference_pipeline():
    print("=== STARTING STAGE 5 INFERENCE & CONFIDENCE VERIFICATION ===")
    
    predictor = CharacterPredictor.get_instance()
    print("  [PASS] CharacterPredictor instance obtained.")

    # -------------------------------------------------------------
    # 1. Test with real EMNIST sample (numpy uint8 array)
    # -------------------------------------------------------------
    print("\n[Test 1] Testing with Real EMNIST Test Sample...")
    _, (X_test, y_test), _ = load_emnist_letters(fix_zero_indexed=True)
    # Pick first 3 test samples
    for i in range(3):
        true_char = CLASS_MAPPING[y_test[i]]
        sample_img = X_test[i]
        res = predictor.predict(sample_img, confidence_threshold=0.70)
        assert res["is_valid"], "EMNIST sample should be valid"
        assert res["status"] in ["success", "uncertain"]
        assert res["prediction"] in [true_char, "Uncertain"]
        assert len(res["top_3"]) == 3
        print(f"  Sample {i+1}: True='{true_char}', Pred='{res['prediction']}', Conf={res['confidence_percent']}%, Status={res['status']}")
    print("  [PASS] Real EMNIST samples successfully processed.")

    # -------------------------------------------------------------
    # 2. Test with RGB, Grayscale, and Bytes Inputs
    # -------------------------------------------------------------
    print("\n[Test 2] Testing RGB, Grayscale, and Raw Bytes Input Formats...")
    rgb_img = draw_test_character("A", size=(100, 100), bg="white", fg="black")
    gray_img = rgb_img.convert("L")
    
    # Create PNG bytes in memory
    buf = io.BytesIO()
    rgb_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    res_rgb = predictor.predict(rgb_img)
    res_gray = predictor.predict(gray_img)
    res_bytes = predictor.predict(png_bytes)

    assert res_rgb["is_valid"] and res_gray["is_valid"] and res_bytes["is_valid"]
    assert res_rgb["raw_prediction"] == "A"
    assert res_gray["raw_prediction"] == "A"
    assert res_bytes["raw_prediction"] == "A"
    print(f"  RGB Input:   Pred='{res_rgb['prediction']}', Conf={res_rgb['confidence_percent']}%")
    print(f"  Gray Input:  Pred='{res_gray['prediction']}', Conf={res_gray['confidence_percent']}%")
    print(f"  Bytes Input: Pred='{res_bytes['prediction']}', Conf={res_bytes['confidence_percent']}%")
    print("  [PASS] All image container formats (PIL RGB, Grayscale, Bytes) supported consistently.")

    # -------------------------------------------------------------
    # 3. Test Polarity Handling (Paper scan vs Dark canvas)
    # -------------------------------------------------------------
    print("\n[Test 3] Testing Polarity Normalization in Inference...")
    paper_a = draw_test_character("A", size=(120, 120), bg="white", fg="black")
    canvas_a = draw_test_character("A", size=(120, 120), bg="black", fg="white")

    res_paper = predictor.predict(paper_a)
    res_canvas = predictor.predict(canvas_a)

    assert res_paper["is_valid"] and res_canvas["is_valid"]
    assert res_paper["raw_prediction"] == "A"
    assert res_canvas["raw_prediction"] == "A"
    print(f"  White Paper / Black Ink: Pred='{res_paper['prediction']}', Conf={res_paper['confidence_percent']}%")
    print(f"  Dark Canvas / White Ink: Pred='{res_canvas['prediction']}', Conf={res_canvas['confidence_percent']}%")
    print("  [PASS] Both polarity orientations correctly classified as 'A'.")

    # -------------------------------------------------------------
    # 4. Test Top-3 Structure & Properties
    # -------------------------------------------------------------
    print("\n[Test 4] Testing Top-3 Structure and Probability Ordering...")
    res_top3 = predictor.predict(rgb_img)
    top_3 = res_top3["top_3"]
    
    assert len(top_3) == 3, f"Expected 3 items in top_3, got {len(top_3)}"
    assert top_3[0]["character"] == res_top3["raw_prediction"], "Rank 1 must match raw prediction"
    assert top_3[0]["probability"] >= top_3[1]["probability"] >= top_3[2]["probability"], "Top-3 must be descending"
    for item in top_3:
        assert item["character"] in CLASS_MAPPING.values(), f"Invalid character: {item['character']}"
        assert 0.0 <= item["probability"] <= 1.0, f"Invalid probability: {item['probability']}"
    
    print("  Top-3 Output:")
    for item in top_3:
        print(f"    Rank {item['rank']}: '{item['character']}' - {item['percentage']}")
    print("  [PASS] Top-3 format, ordering, and character validity verified.")

    # -------------------------------------------------------------
    # 5. Test Real Low-Confidence & Configurable Thresholding
    # -------------------------------------------------------------
    print("\n[Test 5] Testing Low-Confidence & Configurable Thresholding...")
    # 5a. Real ambiguous handwritten test sample (sample index 57 in test set)
    ambiguous_test_img = X_test[57]
    res_ambiguous = predictor.predict(ambiguous_test_img, confidence_threshold=0.70)
    assert res_ambiguous["is_valid"], "Ambiguous test sample is valid image data"
    assert res_ambiguous["confidence"] < 0.70, f"Expected confidence < 0.70, got {res_ambiguous['confidence']}"
    assert res_ambiguous["status"] == "uncertain", f"Expected 'uncertain' status, got {res_ambiguous['status']}"
    assert res_ambiguous["prediction"] == "Uncertain"
    assert "not confident enough" in res_ambiguous["message"]
    print(f"  Real Ambiguous Sample (Index 57): Status='{res_ambiguous['status']}', Raw='{res_ambiguous['raw_prediction']}', Conf={res_ambiguous['confidence_percent']}%, Pred='{res_ambiguous['prediction']}'")

    # 5b. Threshold Gating on clean 'A':
    # Under standard 70% threshold -> 'success' ('A')
    res_clean_70 = predictor.predict(paper_a, confidence_threshold=0.70)
    assert res_clean_70["status"] == "success" and res_clean_70["prediction"] == "A"
    print(f"  Clean 'A' at 70% threshold: Status='{res_clean_70['status']}', Pred='{res_clean_70['prediction']}', Conf={res_clean_70['confidence_percent']}%")

    # Under 99.999% threshold (higher than confidence) -> flips to 'uncertain' ('Uncertain')
    res_clean_99 = predictor.predict(paper_a, confidence_threshold=0.99999)
    assert res_clean_99["status"] == "uncertain" and res_clean_99["prediction"] == "Uncertain"
    print(f"  Clean 'A' at 99.999% threshold: Status='{res_clean_99['status']}', Pred='{res_clean_99['prediction']}' (Threshold gating successfully confirmed)")
    print("  [PASS] Low-confidence detection and configurable threshold gating fully verified.")

    # -------------------------------------------------------------
    # 6. Test Blank and Corrupted Inputs
    # -------------------------------------------------------------
    print("\n[Test 6] Testing Blank and Corrupted Input Defense...")
    blank_white = np.ones((100, 100), dtype=np.uint8) * 255
    blank_black = np.zeros((100, 100), dtype=np.uint8)
    corrupted_bytes = b"not_an_image_data_stream_12345"
    empty_arr = np.array([])

    res_bw = predictor.predict(blank_white)
    res_bb = predictor.predict(blank_black)
    res_corrupt = predictor.predict(corrupted_bytes)
    res_empty = predictor.predict(empty_arr)

    assert not res_bw["is_valid"] and res_bw["status"] == "invalid" and res_bw["prediction"] == "Invalid Input"
    assert not res_bb["is_valid"] and res_bb["status"] == "invalid" and res_bb["prediction"] == "Invalid Input"
    assert not res_corrupt["is_valid"] and res_corrupt["status"] == "invalid"
    assert not res_empty["is_valid"] and res_empty["status"] == "invalid"

    print(f"  Blank White:   Status='{res_bw['status']}', Msg='{res_bw['message']}'")
    print(f"  Blank Black:   Status='{res_bb['status']}', Msg='{res_bb['message']}'")
    print(f"  Corrupt Bytes: Status='{res_corrupt['status']}', Msg='{res_corrupt['message']}'")
    print(f"  Empty Array:   Status='{res_empty['status']}', Msg='{res_empty['message']}'")
    print("  [PASS] Blank and invalid inputs handled safely without crashes.")

    # -------------------------------------------------------------
    # 7. Test Singleton Instance / Fast Repeated Inference
    # -------------------------------------------------------------
    print("\n[Test 7] Testing Predictor Reuse Performance...")
    p2 = CharacterPredictor.get_instance()
    assert p2 is predictor, "Singleton should return identical instance"
    assert p2.model is predictor.model, "Model should not be reloaded"
    
    # Run 10 rapid predictions
    for _ in range(10):
        r = predictor.predict(rgb_img)
        assert r["is_valid"]
    print("  [PASS] 10 consecutive inferences executed rapidly using shared model in memory.")

    print("\n>>> ALL STAGE 5 INFERENCE & CONFIDENCE TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_stage5_inference_pipeline()
