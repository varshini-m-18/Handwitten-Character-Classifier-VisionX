"""Verification script for all 10 Stage 6 UI workflow test scenarios."""
import os
import io
import sys
from PIL import Image
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import CharacterPredictor

def test_ui_scenarios():
    print("=== STARTING STAGE 6 UI WORKFLOW SCENARIO TESTS ===")
    
    predictor = CharacterPredictor.get_instance()
    sample_dir = os.path.join(PROJECT_ROOT, "data", "test_samples")

    # TEST 1: Upload a clear A
    print("\n[UI Test 1] Upload a clear 'A'...")
    img_a = Image.open(os.path.join(sample_dir, "emnist_A.png"))
    res_a = predictor.predict(img_a, confidence_threshold=0.70)
    assert res_a["is_valid"] and res_a["status"] == "success" and res_a["prediction"] == "A"
    print(f"  [PASS] Letter 'A' classified: Pred='{res_a['prediction']}', Conf={res_a['confidence_percent']}%")

    # TEST 2: Upload a clear different letter ('B', 'M', 'Z')
    print("\n[UI Test 2] Upload a clear different letter ('B', 'M', 'Z')...")
    for letter in ["B", "M", "Z"]:
        img_l = Image.open(os.path.join(sample_dir, f"emnist_{letter}.png"))
        res_l = predictor.predict(img_l, confidence_threshold=0.70)
        assert res_l["is_valid"] and res_l["status"] == "success" and res_l["prediction"] == letter
        print(f"  [PASS] Letter '{letter}' classified: Pred='{res_l['prediction']}', Conf={res_l['confidence_percent']}%")

    # TEST 3: Paper-drawn character (Handwritten 'H')
    print("\n[UI Test 3] Drawn/Paper Handwritten 'H'...")
    img_h = Image.open(os.path.join(sample_dir, "paper_handwritten_H.png"))
    res_h = predictor.predict(img_h, confidence_threshold=0.70)
    assert res_h["is_valid"] and res_h["status"] == "success" and res_h["prediction"] == "H"
    print(f"  [PASS] Handwritten 'H' classified: Pred='{res_h['prediction']}', Conf={res_h['confidence_percent']}%")

    # TEST 4: Blank Canvas (Solid black drawing canvas)
    print("\n[UI Test 4] Blank Canvas input...")
    blank_canvas = Image.new("RGB", (280, 280), color="black")
    res_bc = predictor.predict(blank_canvas)
    assert not res_bc["is_valid"] and res_bc["status"] == "invalid"
    print(f"  [PASS] Blank Canvas safely caught: Status='{res_bc['status']}', Msg='{res_bc['message']}'")

    # TEST 5: Blank uploaded image (Solid white paper)
    print("\n[UI Test 5] Blank uploaded image...")
    blank_paper = Image.open(os.path.join(sample_dir, "blank_sample.png"))
    res_bp = predictor.predict(blank_paper)
    assert not res_bp["is_valid"] and res_bp["status"] == "invalid"
    print(f"  [PASS] Blank Upload safely caught: Status='{res_bp['status']}', Msg='{res_bp['message']}'")

    # TEST 6: Low-confidence / ambiguous input
    print("\n[UI Test 6] Low-confidence / Ambiguous input...")
    amb_img = Image.open(os.path.join(sample_dir, "emnist_ambiguous_low_conf.png"))
    res_amb = predictor.predict(amb_img, confidence_threshold=0.70)
    assert res_amb["is_valid"] and res_amb["status"] == "uncertain" and res_amb["prediction"] == "Uncertain"
    assert res_amb["confidence"] < 0.70
    print(f"  [PASS] Ambiguous sample correctly triggered 'Uncertain': Conf={res_amb['confidence_percent']}% < 70%, Raw='{res_amb['raw_prediction']}'")

    # TEST 7: Change confidence threshold slider behavior
    print("\n[UI Test 7] Changing confidence threshold...")
    # At 30% threshold, the ambiguous sample (38.1%) becomes accepted
    res_low_thresh = predictor.predict(amb_img, confidence_threshold=0.30)
    assert res_low_thresh["status"] == "success"
    # At 80% threshold, it is uncertain
    res_high_thresh = predictor.predict(amb_img, confidence_threshold=0.80)
    assert res_high_thresh["status"] == "uncertain"
    print(f"  [PASS] Slider gating dynamic: at 30% threshold -> '{res_low_thresh['status']}', at 80% threshold -> '{res_high_thresh['status']}'")

    # TEST 8: Top-3 predictions appear correctly
    print("\n[UI Test 8] Top-3 predictions formatting...")
    assert len(res_a["top_3"]) == 3
    assert res_a["top_3"][0]["character"] == "A"
    assert res_a["top_3"][0]["probability"] >= res_a["top_3"][1]["probability"] >= res_a["top_3"][2]["probability"]
    print(f"  [PASS] Top-3: {[(item['character'], item['percentage']) for item in res_a['top_3']]}")

    # TEST 9: Preprocessed 28x28 model input
    print("\n[UI Test 9] Preprocessed 28x28 image display shape...")
    vis = res_a["preprocessed_image_28x28"]
    assert vis.shape == (28, 28) and vis.dtype == np.uint8
    print(f"  [PASS] Preprocessed image array shape={vis.shape}, dtype={vis.dtype}, range=[{vis.min()}, {vis.max()}]")

    # TEST 10: Model caching verification
    print("\n[UI Test 10] Model instance caching across runs...")
    p2 = CharacterPredictor.get_instance()
    assert p2 is predictor and p2.model is predictor.model
    print("  [PASS] Model instance cached and reused in memory.")

    print("\n>>> ALL 10 STAGE 6 UI WORKFLOW TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_ui_scenarios()
