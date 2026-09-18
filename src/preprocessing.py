"""Image preprocessing and data augmentation module for handwritten character classification."""
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

def preprocess_dataset_images(images: np.ndarray) -> np.ndarray:
    """
    Preprocess training or testing images from the dataset.
    
    Args:
        images: uint8 numpy array of shape (N, 28, 28) with values in [0, 255].
    
    Returns:
        float32 numpy array of shape (N, 28, 28, 1) normalized to [0.0, 1.0].
    """
    if images.dtype != np.float32:
        norm_images = images.astype(np.float32) / 255.0
    else:
        norm_images = images.copy()
        if norm_images.max() > 1.0:
            norm_images = norm_images / 255.0

    if norm_images.ndim == 3:
        norm_images = np.expand_dims(norm_images, axis=-1)
    
    return norm_images

def check_image_validity(gray_img: np.ndarray, min_std: float = 8.0, min_dynamic_range: int = 25) -> tuple[bool, str]:
    """
    Validate whether an image contains sufficient contrast/strokes or is blank/invalid.
    
    Args:
        gray_img: 2D uint8 numpy array.
        min_std: minimum pixel standard deviation required.
        min_dynamic_range: minimum difference between max and min pixel values.
        
    Returns:
        (is_valid, reason_message)
    """
    if gray_img is None or gray_img.size == 0:
        return False, "Image is empty or unreadable."
    
    dynamic_range = int(gray_img.max()) - int(gray_img.min())
    std_dev = float(np.std(gray_img))
    
    if dynamic_range < min_dynamic_range or std_dev < min_std:
        return False, f"Image appears blank or has insufficient stroke contrast (std: {std_dev:.1f}, range: {dynamic_range})."
    
    return True, "Valid character image."

def center_and_pad_character(char_img: np.ndarray, target_size: tuple[int, int] = (28, 28), box_size: int = 20) -> np.ndarray:
    """
    Center the character inside target_size (default 28x28) using its bounding box,
    scaling the character to fit within box_size (default 20x20) to match standard MNIST/EMNIST formatting.
    
    Args:
        char_img: 2D uint8 array with white character on black background (0=bg, 255=stroke).
        target_size: (height, width) of output canvas.
        box_size: maximum bounding box size inside canvas.
        
    Returns:
        2D uint8 array of shape target_size.
    """
    # Threshold to locate character pixels
    _, thresh = cv2.threshold(char_img, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to direct resize if no contours found
        return cv2.resize(char_img, target_size, interpolation=cv2.INTER_AREA)
    
    # Adaptive stroke preservation: if high-res image has a very thin stroke (<5% non-zero),
    # apply a 3x3 dilation so it does not attenuate/vanish upon downsampling to 20x20
    if max(char_img.shape) > 60:
        stroke_ratio = np.count_nonzero(thresh) / thresh.size
        if stroke_ratio < 0.05:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            char_img = cv2.dilate(char_img, kernel, iterations=1)
    
    # Get bounding box of all contours combined
    x_min, y_min = char_img.shape[1], char_img.shape[0]
    x_max, y_max = 0, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    crop_w = max(1, x_max - x_min)
    crop_h = max(1, y_max - y_min)
    cropped = char_img[y_min:y_max, x_min:x_max]
    
    # Scale proportionally so largest dimension fits into box_size
    scale = box_size / max(crop_w, crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    resized_crop = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create target canvas and center resized crop
    canvas = np.zeros(target_size, dtype=np.uint8)
    pad_top = (target_size[0] - new_h) // 2
    pad_left = (target_size[1] - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_crop
    
    return canvas

def preprocess_user_image(
    image_input,
    target_size: tuple[int, int] = (28, 28),
    normalize: bool = True
) -> tuple[np.ndarray, bool, str, np.ndarray]:
    """
    Full preprocessing pipeline for arbitrary user input (uploads or canvas drawings).
    
    Steps:
    1. Input loading: handles filepath, PIL Image, or numpy array.
    2. Format normalization: handles RGBA alpha blending, RGB to Grayscale.
    3. Blank detection: checks contrast/std before heavy processing.
    4. Polarity normalization: detects dark-on-light vs light-on-dark, inverting to EMNIST format (white stroke on black bg).
    5. Centering & aspect-ratio-preserving padding to 28x28 (standard EMNIST convention).
    6. Normalization: scales pixels to [0.0, 1.0].
    7. Channel expansion: formats to (1, 28, 28, 1) for CNN inference.
    
    Returns:
        (processed_batch, is_valid, status_message, intermediate_uint8_28x28)
        - processed_batch: float32 ndarray of shape (1, target_size[0], target_size[1], 1)
        - is_valid: bool indicating if image contains a clear character
        - status_message: explanation of validity or issues
        - intermediate_uint8_28x28: uint8 ndarray of shape (28, 28) for UI inspection
    """
    # 1. Load input
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            dummy = np.zeros((1, *target_size, 1), dtype=np.float32)
            return dummy, False, f"File not found: {image_input}", np.zeros(target_size, dtype=np.uint8)
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            dummy = np.zeros((1, *target_size, 1), dtype=np.float32)
            return dummy, False, "Input array is empty.", np.zeros(target_size, dtype=np.uint8)
        # Convert numpy to PIL for uniform handling
        if image_input.dtype != np.uint8:
            image_input = np.clip(image_input, 0, 255).astype(np.uint8)
        if image_input.ndim == 2:
            pil_img = Image.fromarray(image_input, mode="L")
        elif image_input.ndim == 3 and image_input.shape[2] == 4:
            pil_img = Image.fromarray(image_input, mode="RGBA")
        elif image_input.ndim == 3 and image_input.shape[2] == 3:
            pil_img = Image.fromarray(image_input, mode="RGB")
        elif image_input.ndim == 3 and image_input.shape[2] == 1:
            pil_img = Image.fromarray(image_input[:, :, 0], mode="L")
        else:
            dummy = np.zeros((1, *target_size, 1), dtype=np.float32)
            return dummy, False, f"Unsupported array shape: {image_input.shape}", np.zeros(target_size, dtype=np.uint8)
    else:
        dummy = np.zeros((1, *target_size, 1), dtype=np.float32)
        return dummy, False, f"Unsupported image input type: {type(image_input)}", np.zeros(target_size, dtype=np.uint8)

    # 2. Handle Alpha channel if present (e.g. transparent canvas drawings)
    if pil_img.mode == "RGBA":
        # Create solid white background for alpha composite
        background = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(background, pil_img)
        gray = np.array(composite.convert("L"))
    else:
        gray = np.array(pil_img.convert("L"))

    # 3. Check for blank or non-contrast images
    is_valid, msg = check_image_validity(gray)
    if not is_valid:
        dummy = np.zeros((1, *target_size, 1), dtype=np.float32)
        return dummy, False, msg, cv2.resize(gray, target_size)

    # 4. Polarity normalization:
    # EMNIST format: Background is ~0 (black), character stroke is ~255 (white).
    # Inspect borders (outer 5%) to determine background brightness.
    h, w = gray.shape
    border_mask = np.ones_like(gray, dtype=bool)
    pad_h, pad_w = max(1, int(h * 0.05)), max(1, int(w * 0.05))
    border_mask[pad_h:h - pad_h, pad_w:w - pad_w] = False
    border_mean = float(np.mean(gray[border_mask]))

    if border_mean > 127:
        # Light background (e.g. paper scan or white canvas) -> Invert
        emnist_style = 255 - gray
    else:
        # Dark background (e.g. black canvas) -> Keep
        emnist_style = gray.copy()

    # 5. Center and pad character to target 28x28
    processed_28x28 = center_and_pad_character(emnist_style, target_size=target_size, box_size=20)

    # Secondary check after centering: ensure character stroke was preserved
    if np.max(processed_28x28) < 40 or np.std(processed_28x28) < 5:
        dummy = np.zeros((1, *target_size, 1), dtype=np.float32)
        return dummy, False, "Character stroke could not be detected after extraction.", processed_28x28

    # 6. Normalize to [0.0, 1.0] float32
    if normalize:
        processed_float = processed_28x28.astype(np.float32) / 255.0
    else:
        processed_float = processed_28x28.astype(np.float32)

    # 7. Channel expansion to batch format (1, 28, 28, 1)
    processed_batch = np.expand_dims(processed_float, axis=(0, -1))

    return processed_batch, True, "Character preprocessed successfully.", processed_28x28

def get_training_augmentation_model() -> keras.Sequential:
    """
    Data augmentation pipeline for CNN training only.
    Applies subtle geometric transformations:
      - RandomRotation: factor 0.08 (~ +-28 deg)
      - RandomTranslation: factor 0.08 (~ +-2.2 pixels)
      - RandomZoom: factor 0.08 (~ +-8%)
    
    Does NOT apply random horizontal flip because letters are orientation-sensitive (e.g. 'B' vs flipped, 'E', 'J').
    """
    return keras.Sequential(
        [
            keras.layers.RandomRotation(factor=0.08, fill_mode="constant", fill_value=0.0, name="aug_rotation"),
            keras.layers.RandomTranslation(
                height_factor=0.08,
                width_factor=0.08,
                fill_mode="constant",
                fill_value=0.0,
                name="aug_translation"
            ),
            keras.layers.RandomZoom(
                height_factor=(-0.08, 0.08),
                width_factor=(-0.08, 0.08),
                fill_mode="constant",
                fill_value=0.0,
                name="aug_zoom"
            ),
        ],
        name="data_augmentation"
    )
