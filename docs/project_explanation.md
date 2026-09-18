# Project Explanation — Handwritten Character Classifier
### IC-09 | VisionX 2026 | Beginner-Friendly Guide

---

## What Problem Does This Project Solve?

Imagine you have a photo of a handwritten letter — say, someone wrote the letter **"A"** on a piece of paper, and you photographed it. A human can read it instantly, but a computer sees only a grid of numbers (pixel values). The computer has no idea that those numbers represent an "A."

This project teaches a computer to look at a picture of a single handwritten uppercase English letter and **correctly identify which letter it is** (A through Z).

This kind of problem is called **Optical Character Recognition (OCR)** — automatically reading characters from images. It has real uses in:
- Scanning handwritten forms and postal letters
- Digitising historical handwritten records
- Assistive technology for visually impaired users

---

## Why Is Preprocessing Needed?

When users send an image to the classifier, it might be:
- A photo from a phone camera (high resolution, colour)
- A scan on a white piece of paper (light background, dark ink)
- A drawing on a black digital canvas (dark background, white ink)
- A tiny 48×48 thumbnail or a large 250×350 photograph

The model was trained on a very specific format: **28×28 pixel grayscale images with a white character on a black background.** If you feed the model a 500×400 colour photo directly, the numbers it sees are completely different from what it learned during training, and it will give wrong answers.

**Preprocessing** is the step that bridges this gap: it transforms whatever the user provides into exactly the 28×28 grayscale white-on-black format the model expects, regardless of the original image's size, colour, or orientation.

---

## Why 28 × 28?

The EMNIST Letters dataset — the collection of real handwritten images used to train this model — stores every image as a **28-pixel-wide by 28-pixel-tall grayscale image**. This is the same format as the famous MNIST digits dataset.

28×28 = 784 pixels. This is:
- Small enough to train efficiently on a laptop CPU
- Large enough to capture the distinctive shape of each letter
- The de facto standard for handwriting benchmarks

Every image the model ever sees during training is 28×28 — so every image sent for prediction must also be 28×28.

---

## What Is a CNN?

**CNN** stands for **Convolutional Neural Network**. It is a type of artificial intelligence model that is especially good at understanding images.

You can think of a neural network as a chain of mathematical machines. Each machine takes numbers in, does some calculation, and passes numbers out to the next machine. By training on many examples, the network learns what calculations to do so that the final output is the right answer.

What makes a **convolutional** neural network special for images is that it uses a technique called **convolution** — explained below — to look at small patches of an image at a time, which is much smarter than looking at all 784 pixels at once.

---

## Why Use a CNN for This Task?

A handwritten "A" can look very different depending on the person who wrote it. Some people write pointed peaks, others write curved ones. A simple approach (like comparing pixel-by-pixel with a template) would fail for most real handwriting.

CNNs solve this because they automatically learn:
- **Edge detectors** — where strokes start and end
- **Shape detectors** — loops, crossbars, diagonal lines
- **Higher-level pattern detectors** — combinations of shapes that spell out a letter

These detectors are learned automatically from thousands of training examples — you do not have to program them by hand.

---

## Convolution, Filters, and Pooling

### Convolution and Filters

A **filter** (also called a kernel) is a small grid of numbers — typically 3×3. The filter slides across the image pixel by pixel, and at each position it multiplies its numbers with the image pixels underneath and sums the results. This produces a single output number for that position.

Doing this for every position in the image produces a new, transformed image called a **feature map**. Different filters detect different things: one filter might respond strongly to vertical edges, another to horizontal edges, another to curves.

In this project:
- **Conv Block 1**: 32 filters of size 3×3 — detects 32 different kinds of low-level features
- **Conv Block 2**: 64 filters of size 3×3 — detects 64 mid-level features from the previous 32
- **Conv Block 3**: 64 filters of size 3×3 — detects 64 higher-level features

### MaxPooling

After each convolutional layer, a **MaxPooling2D (2×2)** layer is applied. It divides the feature map into 2×2 blocks and keeps only the maximum value in each block.

This does two things:
1. **Reduces the spatial size by half** — making the computation faster
2. **Makes the detection more robust** — a feature detected slightly to the left or right still registers

After three rounds of convolution + pooling on a 28×28 image:
- After Pool 1: 14×14 feature maps
- After Pool 2: 7×7 feature maps
- After Pool 3: 3×3 feature maps

---

## Softmax

After all the convolutional processing, the extracted features are flattened into a list of numbers and passed through two Dense (fully connected) layers. The final Dense layer has **26 output units** — one for each letter A through Z.

**Softmax** is the activation function on this final layer. It converts the 26 raw numbers into 26 **probabilities** that all add up to 1.0 (i.e., 100%).

For example, after seeing the image:
```
A: 0.9713  (97.13%)
B: 0.0142  ( 1.42%)
D: 0.0051  ( 0.51%)
... (23 other letters share the remaining ~1%)
```

The letter with the highest probability is the model's best guess.

---

## Confidence

**Confidence** is the probability the model assigns to its top-1 prediction, expressed as a percentage.

- Confidence of **99.99%** for "M" means the model is extremely certain.
- Confidence of **57.53%** means the model is guessing — it leans toward one letter but is not very sure.

Confidence comes directly from the softmax output. It is a measure of how clearly the features in the input image matched what the model learned for that letter.

---

## Confidence Threshold

The system includes a safety mechanism: a **confidence threshold**. By default this is set to **70%**.

- If the top-1 confidence is **≥ 70%** → the system reports the prediction as a confirmed result (`success`).
- If the top-1 confidence is **< 70%** → the system does NOT report a letter. Instead it shows **"Uncertain"** and asks the user to provide a clearer image.

This prevents the system from confidently giving a wrong answer. For example, if someone uploads a photo of a squiggle and the model gives "Q" at 57% confidence, showing "Q" would be misleading. Showing "Uncertain" is more honest.

The threshold is adjustable from **50% to 99%** using the slider in the sidebar. Lowering it makes the system more permissive (more letters get through). Raising it makes it stricter.

---

## Uncertain / Invalid Input Handling

There are three possible outcomes for every input:

### 1. `success`
The preprocessing succeeded, the model ran, and confidence ≥ threshold. The predicted letter is shown.

### 2. `uncertain`
The preprocessing succeeded, the model ran, but confidence < threshold. The model has a candidate letter (shown for reference) but the system does not commit to it.

**Example from Stage 7A testing:**
- An ambiguous EMNIST sample → model predicted "A" at 38.10% → system correctly returned `uncertain`.

### 3. `invalid`
The image was rejected **before** the model even ran, because it contains no usable character:
- A completely white or completely black image
- An image with pixel standard deviation < 8.0 or dynamic range < 25
- Corrupt or unreadable image bytes

**Example from Stage 7A testing:**
- A blank white image → detected by `check_image_validity()` → returned `invalid` with std = 0.0, range = 0.

The `invalid` path skips the neural network entirely, which is both faster and safer.

---

## EMNIST

**EMNIST** stands for **Extended MNIST**. It is a large public dataset of handwritten characters collected by the **National Institute of Standards and Technology (NIST)** in the United States.

The **EMNIST Letters** split used in this project contains:
- **124,800 training images** of handwritten uppercase letters
- **20,800 test images** (exactly 800 per letter A–Z)
- Each image: 28×28 pixels, grayscale, white stroke on black background

The images were contributed by real people writing on paper, then scanned and digitised. This gives the model exposure to a wide variety of real human handwriting styles.

**One quirk:** The EMNIST binary files store images rotated 90° clockwise and horizontally flipped. The `src/dataset.py` loader fixes this automatically by transposing the image axes before use.

---

## How Training Works

Training is the process of making the model learn from examples. Here is what happens step by step:

1. **Load data:** 124,800 labelled training images are loaded from EMNIST. Each image has a known label (e.g., this image is "A").

2. **Split:** 10% of training data (stratified) is held out as a **validation set** to monitor progress. The model never trains on validation data.

3. **Preprocess:** Each image is converted to float32 and normalised to [0.0, 1.0].

4. **Forward pass:** An image is fed through the CNN. The model produces 26 probabilities.

5. **Loss calculation:** The **sparse categorical crossentropy** loss function measures how wrong the prediction was. If the image is "A" and the model said "B" with 95% confidence, the loss is high.

6. **Backpropagation:** The loss is used to calculate how each filter weight and dense layer weight should change to do better next time. The **Adam optimiser** applies these updates with a learning rate of 0.001.

7. **Repeat** for every image in every epoch (one pass through all training data = one epoch).

8. **Early stopping:** After each epoch, validation accuracy is checked. If it stops improving for 3 consecutive epochs, training stops early and the best weights are restored. In this project, training ran for **8 epochs** before stopping (best epoch was Epoch 5 with 93.37% validation accuracy).

9. **Save:** The best model checkpoint is saved to `models/handwritten_character_model.keras`.

---

## What Happens When the User Uploads or Draws an Image

When an image appears in the app (via Upload or the Download-from-canvas workflow):

1. The image is stored in Streamlit's session state so it persists across tab switches.
2. A **preprocessing preview** runs immediately: the image is passed through `preprocess_user_image()` and the resulting 28×28 grayscale tensor is displayed on screen — this lets the user see exactly what the model will see.
3. If the preprocessing detects an invalid image (blank, insufficient contrast), a warning caption is shown under the preview.

---

## What Happens When "Classify Character" Is Clicked

1. Streamlit calls `predictor.predict(active_image, confidence_threshold)`.
2. The `CharacterPredictor` singleton (which already has the trained model in memory) runs `preprocess_user_image()` again to get the clean (1, 28, 28, 1) tensor.
3. If the image is invalid → returns `invalid` response immediately.
4. If valid → the tensor is passed through the CNN using `model(x, training=False)`.
   - `training=False` is critical: it disables the dropout layer and the data augmentation layers that are only active during training.
5. The 26-element softmax output is extracted, clipped to [0.0, 1.0], and re-normalised for numerical safety.
6. The top-1 index and confidence are extracted. The top-3 predictions are sorted descending.
7. If confidence ≥ threshold → `success` result.  
   If confidence < threshold → `uncertain` result.
8. The structured result dictionary is returned to `app.py` and displayed in the UI.

---

## Why Real-World Handwriting Can Differ from EMNIST

EMNIST has certain characteristics that real-world handwriting may not share:

| EMNIST Characteristic | Real-World Variability |
|---|---|
| Carefully isolated single characters | Real images may have adjacent characters, smudging, or background noise |
| Collected from a controlled scanning process | Phone photos may have perspective distortion, shadows, uneven lighting |
| Standardised stroke style | People write letters in many distinct regional styles (e.g., European "1" looks different) |
| Always a single character per image | Real users may accidentally include parts of adjacent letters |
| Mostly upright characters | Real writing can be slanted at large angles |

The preprocessing pipeline handles several of these cases (polarity normalisation, arbitrary resolution, thin-stroke dilation), but there are limits to what preprocessing alone can fix. If an image is fundamentally different from EMNIST samples, the model may be less accurate or return `uncertain`.

---

## What 93.36% Accuracy Actually Means

The model was evaluated on 20,800 completely unseen EMNIST test images — images the model had never seen during training.

**93.36% accuracy means:**  
Out of every 100 test images, the model correctly identified the letter on approximately **93 of them** and made a mistake on **7 of them**.

More precisely:
- 20,800 × 0.9336 ≈ **19,419 correct predictions**
- 20,800 × 0.0664 ≈ **1,381 incorrect predictions**

This is measured across 26 evenly balanced classes (800 images each), so it cannot be inflated by one easy letter dominating the count.

**This does NOT mean:**
- The model will be right 93.36% of the time on your personal handwriting — if your handwriting style is very different from EMNIST samples, accuracy may be lower.
- The model is right 93.36% of the time for every letter. G achieves only 75.50% while M achieves 99.38%.

**The confidence threshold adds an extra safety layer on top of raw accuracy:** the model does not commit to a low-confidence answer, so when it does show a result, users can trust it more.

---

## Project Limitations

| Limitation | Explanation |
|---|---|
| Uppercase A–Z only | Trained on EMNIST Letters which contains only uppercase characters. Lowercase letters, digits, punctuation, and non-English scripts are not supported. |
| One character per image | The model classifies one character at a time. Images with words or sentences are not segmented and will produce incorrect results. |
| Domain shift | EMNIST is digitised pen-on-paper handwriting. Phone camera photos, printed fonts, or highly stylised handwriting may have lower accuracy. |
| G, I, L weaker accuracy | G (75.50%), I (66.13%), and L (81.38%) are the weakest classes due to visual overlap between G/Q and I/L in handwriting. |
| Drawing pad workflow | The HTML5 canvas drawing must be downloaded and then re-uploaded due to Streamlit's component isolation. Direct canvas-to-prediction in a single step is not implemented. |
| Confidence threshold is not per-class | The same 70% threshold applies to all letters, even though some letters (like G and I) are inherently less certain. |

---

*This document is part of Stage 8A — Final Documentation of the IC-09 Handwritten Character Classifier project.*  
*All technical values, architecture details, and evaluation numbers are sourced directly from the project's source code and measured test results.*
