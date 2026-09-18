# ✍️ Handwritten Character Classifier

> **VisionX 2026 — IC-09**  
> AI-powered handwritten English character recognition using a Convolutional Neural Network trained on the EMNIST Letters dataset.

---

## Problem Statement

Handwritten character recognition is a foundational computer-vision task with real-world applications in digitising handwritten documents, postal sorting, form processing, and assistive technology. The challenge lies in handling the natural variation in human handwriting — differences in stroke thickness, rotation, size, and writing style — while correctly distinguishing between visually similar letters (e.g., `I` vs `L`, `G` vs `Q`).

This project addresses that challenge for **26 uppercase English letters (A–Z)** with confidence-gated output so that uncertain or invalid inputs are never silently misclassified.

---

## Objectives

- Train a CNN on the EMNIST Letters benchmark to classify single handwritten uppercase characters A–Z.
- Build a robust preprocessing pipeline that normalises arbitrary user input (scanned images, canvas drawings, photo crops) into the 28×28 grayscale format the model expects.
- Gate predictions by a configurable confidence threshold so low-quality or ambiguous inputs return `Uncertain` rather than a forced incorrect label.
- Deliver the system through a Streamlit web application that supports image upload, an interactive drawing pad, and preset demo samples.

---

## Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.12 |
| Deep Learning | TensorFlow / Keras | ≥ 2.16.0 |
| Numerical Computing | NumPy | ≥ 1.26.0 |
| Image Processing | Pillow (PIL) | ≥ 10.0.0 |
| Image Processing (CV) | OpenCV (`opencv-python`) | ≥ 4.8.0 |
| Visualisation | Matplotlib | ≥ 3.8.0 |
| Evaluation | scikit-learn | ≥ 1.4.0 |
| Web UI | Streamlit | ≥ 1.35.0 |
| Drawing Pad | streamlit-drawable-canvas | ≥ 0.9.3 |

---

## EMNIST Dataset

| Property | Value |
|---|---|
| Full Name | EMNIST Letters (Extended MNIST — NIST Special Database 19) |
| Source | Hugging Face mirror: `Royc30ne/emnist-letters` |
| Classes | 26 (uppercase A–Z) |
| Training images | 124,800 |
| Test images | 20,800 |
| Images per class (test) | 800 (perfectly balanced) |
| Image format | 28 × 28 grayscale, uint8, white stroke on black background |
| Label encoding | Raw EMNIST labels 1–26 re-indexed to 0–25 (0 = A, ..., 25 = Z) |

> **Note:** EMNIST binary files store images rotated 90° clockwise and horizontally flipped. `src/dataset.py` corrects this during loading via `np.transpose(data, (0, 2, 1))`.

---

## System Architecture & Pipeline

```
User Input (file path / bytes / PIL Image / NumPy ndarray)
          |
          v
+-----------------------------------+
|  src/preprocessing.py             |
|  preprocess_user_image()          |
|  1. Load & decode input           |
|  2. Convert to grayscale (L)      |
|  3. RGBA alpha-composite          |
|  4. Blank / contrast check        |
|     (std < 8.0 or range < 25)    |
|  5. Polarity normalisation        |
|     (border-mean > 127 -> invert  |
|     to EMNIST white-on-black)     |
|  6. Bounding-box crop + centre    |
|     into 28x28 (box_size=20)     |
|  7. Thin-stroke dilation guard    |
|  8. Normalise to float32 [0,1]   |
|  9. Expand to (1, 28, 28, 1)     |
+----------------+------------------+
                 |
                 v
+-----------------------------------+
|  src/model.py / .keras file       |
|  CNN Forward Pass                 |
|  (training=False, no augment)     |
|  Output: softmax over 26 classes  |
+----------------+------------------+
                 |
                 v
+-----------------------------------+
|  src/prediction.py                |
|  CharacterPredictor.predict()     |
|  - top-1 argmax + confidence      |
|  - top-3 sorted descending        |
|  - threshold gate (default 70%)   |
|  Status: success / uncertain /    |
|          invalid                  |
+----------------+------------------+
                 |
                 v
           Streamlit UI
           app/app.py
```

---

## Preprocessing Pipeline (Detail)

Implemented in `src/preprocessing.py`:

| Step | Function | Detail |
|---|---|---|
| Input loading | `preprocess_user_image()` | Accepts `str` (path), `bytes`, `io.BytesIO`, `PIL.Image` (L/RGB/RGBA), `np.ndarray` (2D/3D) |
| Format normalisation | Inside `preprocess_user_image()` | RGBA → alpha-composite on white → grayscale; RGB → grayscale |
| Blank detection | `check_image_validity()` | Rejects if pixel std < 8.0 or dynamic range < 25 |
| Polarity normalisation | Inside `preprocess_user_image()` | Inspects outer 5% border mean; if > 127 (light background) → inverts to EMNIST format |
| Centering | `center_and_pad_character()` | Finds bounding box of all contours, scales to fit in 20×20, centres on 28×28 canvas |
| Thin-stroke guard | Inside `center_and_pad_character()` | If input > 60px and stroke ratio < 5%, applies 3×3 ellipse dilation before downsampling |
| Normalisation | Inside `preprocess_user_image()` | Divides by 255.0 → float32 [0.0, 1.0] |
| Shape expansion | Inside `preprocess_user_image()` | `np.expand_dims` → (1, 28, 28, 1) batch tensor |

---

## CNN Architecture

Defined in `src/model.py` — `build_character_cnn()`:

```
Input: (28, 28, 1)
|
+-- [Training-only augmentation -- bypassed during inference]
|   +-- RandomRotation   factor=0.06 (~+-21.6 deg)
|   +-- RandomTranslation height_factor=0.06, width_factor=0.06
|   +-- RandomZoom       height_factor=(-0.06, 0.06), width_factor=(-0.06, 0.06)
|
+-- Conv Block 1
|   +-- Conv2D  32 filters, 3x3, padding=same, activation=relu  [conv1]
|   +-- BatchNormalization                                        [bn1]
|   +-- MaxPooling2D  2x2                                         [pool1]
|
+-- Conv Block 2
|   +-- Conv2D  64 filters, 3x3, padding=same, activation=relu  [conv2]
|   +-- BatchNormalization                                        [bn2]
|   +-- MaxPooling2D  2x2                                         [pool2]
|
+-- Conv Block 3
|   +-- Conv2D  64 filters, 3x3, padding=same, activation=relu  [conv3]
|   +-- BatchNormalization                                        [bn3]
|   +-- MaxPooling2D  2x2                                         [pool3]
|
+-- Flatten                                                       [flatten]
+-- Dense  128 units, activation=relu                             [dense1]
+-- Dropout  rate=0.35                                            [dropout]
+-- Dense  26 units, activation=softmax                           [output_probabilities]

Compiled with: Adam (lr=0.001), loss=sparse_categorical_crossentropy, metrics=[accuracy]
```

---

## Training Setup

Training script: `training/train.py`

| Hyperparameter | Value |
|---|---|
| Epochs (max) | 12 |
| Batch size | 128 |
| Initial learning rate | 0.001 (Adam) |
| Validation split | 10% of training data (stratified) |
| Random seed | 42 |
| Early stopping | monitors `val_accuracy`, patience=3, restores best weights |
| Model checkpoint | saves best `val_accuracy` epoch only |
| LR reduction | `ReduceLROnPlateau` on `val_loss`, factor=0.5, patience=2, min_lr=1e-5 |

**Training history (8 epochs completed — early stopping triggered):**

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|---|---|---|---|---|
| 1 | 72.15% | 84.72% | 0.9122 | 0.5922 |
| 2 | 87.10% | 91.36% | 0.3972 | 0.2609 |
| 3 | 89.34% | 91.53% | 0.3290 | 0.2582 |
| 4 | 90.32% | 90.58% | 0.2933 | 0.2887 |
| 5 | 91.08% | **93.37%** | 0.2726 | 0.1967 |
| 6 | 91.48% | 90.58% | 0.2599 | 0.2918 |
| 7 | 91.80% | 91.75% | 0.2496 | 0.2531 |
| 8 | 92.53% | 92.08% | 0.2243 | 0.2412 |

Best epoch: **Epoch 5** (val_accuracy = 93.37%). That checkpoint is the saved model.

---

## Measured Evaluation Results

Evaluated on the full, untouched EMNIST Letters test set (20,800 images, 800 per class).

| Metric | Value |
|---|---|
| **Test Accuracy** | **93.36%** |
| Test Loss | 0.1997 |
| Macro Precision | 93.52% |
| Macro Recall | 93.36% |
| Macro F1 | 93.34% |
| Weighted F1 | 93.34% |
| Test samples | 20,800 |

**Per-class accuracy:**

| Letter | Accuracy | Letter | Accuracy |
|---|---|---|---|
| A | 94.50% | N | 92.13% |
| B | 97.00% | O | 94.75% |
| C | 95.25% | P | 98.25% |
| D | 96.88% | Q | 92.25% |
| E | 95.63% | R | 91.13% |
| F | 97.63% | S | 97.88% |
| G | 75.50% (lowest) | T | 97.75% |
| H | 96.88% | U | 95.25% |
| I | 66.13% (lowest) | V | 89.38% |
| J | 93.50% | W | 98.63% |
| K | 98.50% | X | 96.00% |
| L | 81.38% | Y | 96.50% |
| M | 99.38% | Z | 99.38% |

> Letters G, I, and L have the lowest per-class accuracy due to visual similarity with other characters (G/Q confusion, I/L confusion).

---

## Confidence Handling

Implemented in `src/prediction.py`:

| Status | Condition | What the user sees |
|---|---|---|
| `success` | top-1 confidence >= threshold | Predicted letter + confidence % + top-3 |
| `uncertain` | top-1 confidence < threshold | `?` badge + candidate letter + message to upload clearer image |
| `invalid` | blank image, insufficient contrast, or corrupt bytes | Error message, no prediction attempted |

- **Default threshold:** 70% (configurable 50–99% in the UI sidebar slider)
- The threshold applies **after** model inference — a low score means the model is unsure; the system refuses to force an incorrect label.
- Top-3 predictions are always sorted descending by probability and displayed for both `success` and `uncertain` status.
- The singleton pattern (`CharacterPredictor._instance`) ensures the model is loaded into memory **once** per process.

---

## Streamlit UI Features

Application entry point: `app/app.py`

| Feature | Detail |
|---|---|
| **Upload Image** tab | Accepts PNG, JPG, JPEG; loaded as PIL Image |
| **Interactive Drawing Pad** tab | 280×280 HTML5 canvas (white ink on black, stroke width=18); Download → Upload workflow |
| **Quick Demo Presets** tab | 6 preset buttons: real EMNIST samples A, B, M, Z; ambiguous (low confidence); blank (invalid) |
| **Preprocessing preview** | Shows the 28×28 centred grayscale tensor the model actually receives |
| **Classify Character button** | Triggers full inference pipeline |
| **Result display** | Large character badge (success), `?` badge (uncertain), or error card (invalid) |
| **Confidence progress bar** | Visual confidence meter |
| **Top-3 breakdown** | Character + probability bar for each of the top 3 candidates |
| **All 26 probabilities chart** | Expandable bar chart of the full softmax distribution |
| **Sidebar confidence slider** | Adjustable threshold 50–99% (default 70%) |
| **Cached model loading** | `@st.cache_resource` ensures the model loads once per Streamlit session |

---

## Stage 7A Testing Results

Comprehensive end-to-end test matrix — all **24/24 scenarios passed**.

| # | Scenario | Result | Confidence |
|---|---|---|---|
| 1 | Clear 'A' (EMNIST sample) | A | 97.13% |
| 2 | Clear 'B' (EMNIST sample) | B | 99.76% |
| 3 | Clear 'M' (EMNIST sample) | M | 99.99% |
| 4 | Thick-stroke 'A' (16px) | A | 99.97% |
| 5 | Thin-stroke 'A' (3px) | A | 92.51% |
| 6 | Rotated 'A' (+15 deg) | A | 97.40% |
| 7 | Blurry 'A' (Gaussian) | A | 99.87% |
| 8 | Blank white image | Invalid (correctly rejected) | 0% |
| 9 | Blank black image | Invalid (correctly rejected) | 0% |
| 10 | Non-character / scribble | Uncertain (57.53% < 70%) | 57.53% |
| 11 | Low-confidence ambiguous sample | Uncertain (38.10% < 70%) | 38.10% |
| 12 | PNG format bytes | A | 99.97% |
| 13 | JPEG format bytes | A | 99.97% |
| 14 | Grayscale (PIL Mode L) | A | 99.97% |
| 15 | RGB (PIL Mode RGB) | A | 99.97% |
| 16 | Multiple resolutions (48x48, 120x120, 250x350) | A | 98.01% |
| 17 | White background / black ink | A | 99.88% |
| 18 | Black background / white ink | A | 99.88% |
| 19 | Drawing-pad (simulated canvas) | B | 99.96% |
| 20 | Confidence threshold gating | Uncertain as expected | 38.10% |
| 21 | Top-3 descending order | Verified | 97.13% |
| 22 | 28x28 preprocessed tensor input | A | 97.13% |
| 23 | 25 consecutive predictions (singleton stability) | B (consistent) | 99.76% |
| 24 | Application restart / model reloading | M | 99.99% |

**Zero bugs found. Zero code changes required.**

---

## Installation

### Prerequisites

- Python 3.12
- `pip`

### Steps

```bash
# 1. Enter the project directory
cd vision-ic_hack

# 2. (Recommended) Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

The trained model (`models/handwritten_character_model.keras`) is included. No retraining is required.

---

## How to Run the Application

```bash
streamlit run app/app.py
```

Streamlit will open the application at `http://localhost:8501` in your default browser.

---

## Optional: Re-train the Model

```bash
python training/train.py

# With custom hyperparameters:
python training/train.py --epochs 12 --batch_size 128 --learning_rate 0.001 --patience 3
```

## Optional: Re-evaluate the Model

```bash
python evaluation/evaluate.py
```

---

## Project Structure

```
vision-ic_hack/
|
+-- app/
|   +-- app.py                        # Streamlit web application
|
+-- data/
|   +-- raw/                          # Downloaded EMNIST binary (.gz) files
|   +-- test_samples/                 # Pre-generated demo PNG images
|
+-- evaluation/
|   +-- evaluate.py                   # Evaluation script
|
+-- models/
|   +-- handwritten_character_model.keras  # Trained CNN (1.6 MB)
|
+-- results/
|   +-- classification_report.txt     # Per-class precision/recall/F1
|   +-- confusion_matrix.png          # 26x26 confusion matrix heatmap
|   +-- dataset_samples.png           # Sample visualisation grid (A-Z)
|   +-- evaluation_metrics.json       # Machine-readable evaluation metrics
|   +-- preprocessing_samples.png     # Preprocessing pipeline visualisation
|   +-- stage7a_test_report.json      # Stage 7A 24-test machine-readable report
|   +-- training_history.json         # Per-epoch accuracy and loss
|   +-- training_history.png          # Training curve plots
|
+-- src/
|   +-- __init__.py
|   +-- dataset.py                    # EMNIST download, IDX parsing, label re-indexing
|   +-- model.py                      # CNN architecture and model loader
|   +-- prediction.py                 # CharacterPredictor singleton, confidence gating
|   +-- preprocessing.py             # User-image preprocessing pipeline
|
+-- tests/
|   +-- generate_demo_samples.py      # Generates data/test_samples/ preset PNG files
|   +-- test_dataset.py
|   +-- test_environment.py
|   +-- test_model.py
|   +-- test_prediction.py
|   +-- test_preprocessing.py
|   +-- test_stage7a_comprehensive.py # Stage 7A 24-scenario test matrix
|   +-- test_ui_workflow.py
|
+-- training/
|   +-- train.py                      # Training script with argparse CLI
|
+-- .gitignore
+-- requirements.txt
+-- README.md
```

---

## Example Usage

### Programmatic

```python
from src.prediction import predict_character

# From a file path
result = predict_character("path/to/letter_A.png")
print(result["prediction"])          # 'A'
print(result["confidence_percent"])  # e.g. 97.13

# From a PIL Image
from PIL import Image
img = Image.open("letter.png")
result = predict_character(img, confidence_threshold=0.80)

# From raw bytes
with open("letter.png", "rb") as f:
    result = predict_character(f.read())

# Result structure:
# {
#   "is_valid": True,
#   "status": "success",             # "success" | "uncertain" | "invalid"
#   "prediction": "A",               # final user-facing prediction
#   "raw_prediction": "A",           # top-1 regardless of threshold
#   "confidence": 0.9713,
#   "confidence_percent": 97.13,
#   "top_3": [...],                  # list of 3 dicts: rank, character, probability, percentage
#   "all_probabilities": [...],      # list of 26 floats (softmax output)
#   "threshold": 0.70,
#   "message": "Character identified as 'A' with 97.13% confidence.",
#   "preprocessed_image_28x28": <numpy uint8 28x28 array>
# }
```

---

## Limitations

- **Uppercase only:** Trained on EMNIST Letters (uppercase A–Z only). Lowercase, digits, and symbols are not recognised.
- **Single character per image:** One character per input image; words or multi-character images are not supported.
- **Domain shift:** EMNIST contains carefully digitised samples. Real-world photos, low-contrast scans, or highly stylised handwriting may reduce accuracy.
- **Visually ambiguous letters:** G, I, and L have the lowest per-class accuracy (75.50%, 66.13%, 81.38%) due to inherent visual similarity between these characters.
- **Drawing pad workflow:** Requires a Download → Upload step because Streamlit's iframe isolation prevents direct pixel transfer from the HTML5 canvas to the Python backend.
- **No GPU requirement:** The model runs on CPU, but GPU accelerates inference when available.

---

## Future Improvements

- Extend to lowercase letters using EMNIST ByClass dataset.
- Add digit recognition (0–9) using EMNIST Digits or MNIST.
- Implement real-time direct canvas-to-prediction without the Download/Upload step.
- Improve G, I, and L accuracy through targeted augmentation or class weighting.
- Add word-level recognition by segmenting multi-character inputs.
- Package as a Docker container for one-command deployment.

---

## Project Information

| Field | Detail |
|---|---|
| Project | IC-09 — Handwritten Character Classifier |
| Event | VisionX 2026 Hackathon |
| Stage completed | Stage 8A (Final Documentation) |
| Model file | `models/handwritten_character_model.keras` |
| Test accuracy | 93.36% on 20,800 EMNIST Letters test images |
| All end-to-end tests | 24/24 Stage 7A scenarios passed |
