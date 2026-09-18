"""Comprehensive Stage 7A End-to-End Test Suite covering all 24 test matrix scenarios."""
import os
import io
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import load_emnist_letters
from src.prediction import CharacterPredictor
from src.model import CLASS_MAPPING

def draw_letter(char="A", size=(140, 140), stroke_width=8, bg="white", fg="black", rotate=0, blur=0) -> Image.Image:
    """Helper to draw customized test characters with control over stroke, rotation, and blur."""
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)
    w, h = size
    pad = int(w * 0.22)
    
    if char == "A":
        draw.line([(pad, h - pad), (w // 2, pad)], fill=fg, width=stroke_width)
        draw.line([(w // 2, pad), (w - pad, h - pad)], fill=fg, width=stroke_width)
        draw.line([(int(w * 0.35), int(h * 0.65)), (int(w * 0.65), int(h * 0.65))], fill=fg, width=stroke_width)
    elif char == "B":
        draw.line([(pad, pad), (pad, h - pad)], fill=fg, width=stroke_width)
        draw.arc([pad, pad, int(w * 0.8), int(h * 0.5)], start=270, end=90, fill=fg, width=stroke_width)
        draw.arc([pad, int(h * 0.5), int(w * 0.8), h - pad], start=270, end=90, fill=fg, width=stroke_width)
    elif char == "M":
        draw.line([(pad, h - pad), (pad, pad)], fill=fg, width=stroke_width)
        draw.line([(pad, pad), (w // 2, int(h * 0.65))], fill=fg, width=stroke_width)
        draw.line([(w // 2, int(h * 0.65)), (w - pad, pad)], fill=fg, width=stroke_width)
        draw.line([(w - pad, pad), (w - pad, h - pad)], fill=fg, width=stroke_width)
    elif char == "squiggle":
        # Random non-character scribble
        draw.line([(pad, pad), (w - pad, h - pad)], fill=fg, width=stroke_width)
        draw.line([(w - pad, pad), (pad, h - pad)], fill=fg, width=stroke_width)
        draw.ellipse([pad + 10, pad + 10, w - pad - 10, h - pad - 10], outline=fg, width=stroke_width)

    if rotate != 0:
        # Rotate with background color fill
        img = img.rotate(rotate, fillcolor=bg, resample=Image.Resampling.BICUBIC)
    
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
        
    return img

def run_stage7a_tests():
    print("=" * 70)
    print("STAGE 7A — COMPREHENSIVE END-TO-END TEST MATRIX (24 TEST CASES)")
    print("=" * 70)

    predictor = CharacterPredictor.get_instance()
    _, (X_test, y_test), _ = load_emnist_letters(fix_zero_indexed=True)
    
    results = []

    def record_test(test_id, description, input_type, res, expected_behavior):
        status = res.get("status")
        pred = res.get("prediction")
        conf = res.get("confidence_percent", 0.0)
        raw_pred = res.get("raw_prediction")
        is_reasonable = expected_behavior(res)
        
        record = {
            "test_id": int(test_id),
            "description": str(description),
            "input_type": str(input_type),
            "predicted_character": str(pred) if pred is not None else None,
            "raw_prediction": str(raw_pred) if raw_pred is not None else None,
            "confidence_percent": float(conf),
            "status": str(status),
            "behavior_reasonable": bool(is_reasonable),
            "message": str(res.get("message"))
        }
        results.append(record)
        pass_tag = "[PASS]" if is_reasonable else "[FAIL]"
        print(f"{pass_tag} Test {test_id:02d}: {description}")
        print(f"       Pred: '{pred}', Conf: {conf:.2f}%, Status: '{status}'")
        return bool(is_reasonable)

    # -----------------------------------------------------------------
    # CORE TEST MATRIX (1-11)
    # -----------------------------------------------------------------
    # 1. Clear handwritten A (EMNIST test set sample 0)
    img_a = X_test[0]
    res1 = predictor.predict(img_a, confidence_threshold=0.70)
    record_test(1, "Clear handwritten 'A' (EMNIST)", "EMNIST uint8 ndarray", res1,
                lambda r: r["status"] == "success" and r["prediction"] == "A" and r["confidence"] >= 0.70)

    # 2. Clear handwritten B
    b_indices = np.where(y_test == 1)[0]
    img_b = X_test[b_indices[0]]
    res2 = predictor.predict(img_b, confidence_threshold=0.70)
    record_test(2, "Clear handwritten 'B' (EMNIST)", "EMNIST uint8 ndarray", res2,
                lambda r: r["status"] == "success" and r["prediction"] == "B")

    # 3. Clear handwritten M
    m_indices = np.where(y_test == 12)[0]
    img_m = X_test[m_indices[0]]
    res3 = predictor.predict(img_m, confidence_threshold=0.70)
    record_test(3, "Clear handwritten 'M' (EMNIST)", "EMNIST uint8 ndarray", res3,
                lambda r: r["status"] == "success" and r["prediction"] == "M")

    # 4. Thick-stroke character (A with stroke width = 16)
    img_thick = draw_letter("A", stroke_width=16)
    res4 = predictor.predict(img_thick, confidence_threshold=0.70)
    record_test(4, "Thick-stroke character 'A'", "Synthetic PIL Image (16px stroke)", res4,
                lambda r: r["status"] == "success" and r["prediction"] == "A")

    # 5. Thin-stroke character (A with stroke width = 3)
    img_thin = draw_letter("A", stroke_width=3)
    res5 = predictor.predict(img_thin, confidence_threshold=0.70)
    record_test(5, "Thin-stroke character 'A'", "Synthetic PIL Image (3px stroke)", res5,
                lambda r: r["status"] == "success" and r["prediction"] == "A")

    # 6. Slightly rotated character (A rotated 15 deg)
    img_rot = draw_letter("A", stroke_width=8, rotate=15)
    res6 = predictor.predict(img_rot, confidence_threshold=0.70)
    record_test(6, "Slightly rotated character 'A' (+15 deg)", "Rotated PIL Image", res6,
                lambda r: r["status"] == "success" and r["prediction"] == "A")

    # 7. Slightly blurry character (A with Gaussian blur)
    img_blur = draw_letter("A", stroke_width=8, blur=1.5)
    res7 = predictor.predict(img_blur, confidence_threshold=0.70)
    record_test(7, "Slightly blurry character 'A'", "Blurred PIL Image", res7,
                lambda r: r["status"] == "success" and r["prediction"] == "A")

    # 8. Blank white image
    img_blank_w = Image.new("RGB", (120, 120), color="white")
    res8 = predictor.predict(img_blank_w)
    record_test(8, "Blank white image", "Solid white PIL Image", res8,
                lambda r: not r["is_valid"] and r["status"] == "invalid")

    # 9. Blank black image
    img_blank_b = Image.new("RGB", (120, 120), color="black")
    res9 = predictor.predict(img_blank_b)
    record_test(9, "Blank black image", "Solid black PIL Image", res9,
                lambda r: not r["is_valid"] and r["status"] == "invalid")

    # 10. Non-character input (complex squiggle)
    img_squiggle = draw_letter("squiggle", stroke_width=6)
    res10 = predictor.predict(img_squiggle, confidence_threshold=0.70)
    # Should either be marked uncertain or have low confidence / safe return without crash
    record_test(10, "Non-character input (Scribble/Symbol)", "Non-letter glyph PIL Image", res10,
                lambda r: r["status"] in ["uncertain", "success"] and len(r["top_3"]) == 3)

    # 11. Low-confidence ambiguous character (EMNIST test index 57)
    img_amb = X_test[57]
    res11 = predictor.predict(img_amb, confidence_threshold=0.70)
    record_test(11, "Low-confidence ambiguous character (Sample 57)", "Ambiguous EMNIST sample", res11,
                lambda r: r["status"] == "uncertain" and r["prediction"] == "Uncertain" and r["confidence"] < 0.70)

    # -----------------------------------------------------------------
    # ADDITIONAL SYSTEM & CONTAINER TESTS (12-24)
    # -----------------------------------------------------------------
    # 12. Upload PNG format
    buf_png = io.BytesIO()
    img_thick.save(buf_png, format="PNG")
    res12 = predictor.predict(buf_png.getvalue())
    record_test(12, "Upload PNG format bytes", "PNG raw bytes", res12,
                lambda r: r["status"] == "success" and r["prediction"] == "A")

    # 13. Upload JPG/JPEG format
    buf_jpg = io.BytesIO()
    img_thick.save(buf_jpg, format="JPEG")
    res13 = predictor.predict(buf_jpg.getvalue())
    record_test(13, "Upload JPEG format bytes", "JPEG raw bytes", res13,
                lambda r: r["status"] == "success" and r["prediction"] == "A")

    # 14. Grayscale input
    img_gray = img_thick.convert("L")
    res14 = predictor.predict(img_gray)
    record_test(14, "Grayscale input (Mode 'L')", "Grayscale PIL Image", res14,
                lambda r: r["status"] == "success" and r["prediction"] == "A")

    # 15. RGB input
    img_rgb = img_thick.convert("RGB")
    res15 = predictor.predict(img_rgb)
    record_test(15, "RGB input (Mode 'RGB')", "RGB PIL Image", res15,
                lambda r: r["status"] == "success" and r["prediction"] == "A")

    # 16. Different resolutions (48x48, 120x120, 250x350)
    res16_passed = True
    for sz in [(48, 48), (120, 120), (250, 350)]:
        custom_sz = draw_letter("A", size=sz)
        r = predictor.predict(custom_sz)
        if not (r["status"] == "success" and r["prediction"] == "A"):
            res16_passed = False
    res16 = predictor.predict(draw_letter("A", size=(250, 350)))
    record_test(16, "Different resolutions (48x48, 120x120, 250x350)", "Various image sizes", res16,
                lambda r: res16_passed)

    # 17. White-background / Black-ink input
    img_w_bg = draw_letter("A", bg="white", fg="black")
    res17 = predictor.predict(img_w_bg)
    record_test(17, "White-background / Black-ink input", "Light paper scan", res17,
                lambda r: r["status"] == "success" and r["prediction"] == "A")

    # 18. Black-background / White-ink input
    img_b_bg = draw_letter("A", bg="black", fg="white")
    res18 = predictor.predict(img_b_bg)
    record_test(18, "Black-background / White-ink input", "Dark canvas sketch", res18,
                lambda r: r["status"] == "success" and r["prediction"] == "A")

    # 19. Drawing-pad input (simulated canvas)
    drawn_canvas = draw_letter("B", bg="black", fg="white", stroke_width=12)
    res19 = predictor.predict(drawn_canvas)
    record_test(19, "Drawing-pad input (Simulated Canvas)", "Canvas Drawing Image", res19,
                lambda r: r["status"] == "success" and r["prediction"] == "B")

    # 20. Changing confidence threshold (50%, 70%, 90%, 99.9%)
    t_50 = predictor.predict(img_amb, confidence_threshold=0.50)["status"]  # conf is ~38% -> uncertain
    t_30 = predictor.predict(img_amb, confidence_threshold=0.30)["status"]  # conf is ~38% -> success
    t_99 = predictor.predict(img_a, confidence_threshold=0.99999)["status"] # flips 'A' to uncertain
    res20_passed = (t_50 == "uncertain" and t_30 == "success" and t_99 == "uncertain")
    res20 = predictor.predict(img_amb, confidence_threshold=0.70)
    record_test(20, "Changing confidence threshold gating", "Gating across thresholds", res20,
                lambda r: res20_passed)

    # 21. Top-3 predictions properties
    res21 = predictor.predict(img_a)
    top_3 = res21["top_3"]
    t3_valid = (
        len(top_3) == 3 and
        top_3[0]["character"] == "A" and
        top_3[0]["probability"] >= top_3[1]["probability"] >= top_3[2]["probability"] and
        all(0.0 <= x["probability"] <= 1.0 for x in top_3)
    )
    record_test(21, "Top-3 predictions descending ordering", "Probability ranking", res21,
                lambda r: t3_valid)

    # 22. Preprocessed 28x28 preview
    preview = res21["preprocessed_image_28x28"]
    p_valid = (preview.shape == (28, 28) and preview.dtype == np.uint8 and 0 <= preview.min() and preview.max() <= 255)
    record_test(22, "Preprocessed 28x28 preview tensor", "uint8 28x28 ndarray", res21,
                lambda r: p_valid)

    # 23. Multiple consecutive predictions (reusing loaded model)
    consec_passed = True
    for _ in range(25):
        rc = predictor.predict(img_b)
        if rc["prediction"] != "B":
            consec_passed = False
    record_test(23, "Multiple consecutive predictions (25 runs)", "Memory / Singleton stability", res2,
                lambda r: consec_passed)

    # 24. Application restart and prediction again
    p_fresh = CharacterPredictor(os.path.join(PROJECT_ROOT, "models", "handwritten_character_model.keras"))
    res24 = p_fresh.predict(img_m)
    record_test(24, "Application restart & model reloading", "Fresh predictor instance", res24,
                lambda r: r["status"] == "success" and r["prediction"] == "M")

    # Check that all 24 tests passed
    all_passed = bool(all(r["behavior_reasonable"] for r in results))
    print("\n" + "=" * 70)
    if all_passed:
        print(">>> ALL 24 STAGE 7A TEST MATRIX SCENARIOS PASSED WITH ZERO FAILURES! <<<")
    else:
        failed_tests = [r["test_id"] for r in results if not r["behavior_reasonable"]]
        print(f">>> WARNING: FAILED TESTS DETECTED: {failed_tests} <<<")
    print("=" * 70)

    # Save detailed JSON report
    report_path = os.path.join(PROJECT_ROOT, "results", "stage7a_test_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "total_tests": len(results),
            "all_passed": bool(all_passed),
            "test_results": results
        }, f, indent=2)
    print(f"Saved comprehensive Stage 7A test report to: {report_path}")

    return all_passed

if __name__ == "__main__":
    success = run_stage7a_tests()
    sys.exit(0 if success else 1)
