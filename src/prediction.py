"""End-to-end inference and prediction pipeline with low-confidence and invalid-input handling."""
import os
import io
import sys
from typing import Union, Optional, Any
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import load_trained_model, CLASS_MAPPING, NUM_CLASSES
from src.preprocessing import preprocess_user_image

DEFAULT_CONFIDENCE_THRESHOLD = 0.70  # 70% threshold

class CharacterPredictor:
    """
    Singleton-style predictor that loads the trained CNN model once and serves
    fast, consistent predictions with confidence thresholding and invalid-input guarding.
    """
    _instance: Optional["CharacterPredictor"] = None
    _model: Optional[keras.Model] = None

    def __init__(self, model_path: Optional[str] = None):
        if CharacterPredictor._model is None:
            if model_path is None:
                model_path = os.path.join(PROJECT_ROOT, "models", "handwritten_character_model.keras")
            CharacterPredictor._model = load_trained_model(model_path)
        self.model = CharacterPredictor._model

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "CharacterPredictor":
        """Get or initialize the shared predictor instance."""
        if cls._instance is None:
            cls._instance = CharacterPredictor(model_path)
        return cls._instance

    def predict(
        self,
        image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> dict[str, Any]:
        """
        Run the complete inference pipeline on an input image.
        
        Args:
            image_input: File path (str), raw bytes (bytes/BytesIO), PIL Image, or NumPy ndarray.
            confidence_threshold: Float between 0.0 and 1.0. If top confidence is below this
                                 value, the prediction is flagged as 'Uncertain'.
                                 
        Returns:
            dict containing:
              - 'is_valid': bool, False if image is blank or corrupted
              - 'status': 'success' | 'uncertain' | 'invalid'
              - 'prediction': 'A'-'Z' if success, 'Uncertain' if low confidence, 'Invalid Input' if blank
              - 'raw_prediction': Top predicted character regardless of threshold
              - 'confidence': Top probability (float in [0.0, 1.0])
              - 'confidence_percent': Top probability formatted as percentage float (0.0 to 100.0)
              - 'top_3': list of top 3 predictions [{'character': str, 'probability': float, 'percentage': str}]
              - 'all_probabilities': list of all 26 probabilities (for graphing/debug)
              - 'threshold': confidence_threshold applied
              - 'message': user-facing descriptive message
              - 'preprocessed_image_28x28': uint8 ndarray of shape (28, 28) for UI inspection
        """
        # 1. Normalize bytes input if needed
        if isinstance(image_input, (bytes, bytearray)):
            try:
                image_input = Image.open(io.BytesIO(image_input))
            except Exception as e:
                dummy_vis = np.zeros((28, 28), dtype=np.uint8)
                return self._build_invalid_response(
                    f"Corrupted or unrecognized image bytes: {str(e)}",
                    dummy_vis,
                    confidence_threshold
                )
        elif isinstance(image_input, io.BytesIO):
            try:
                image_input = Image.open(image_input)
            except Exception as e:
                dummy_vis = np.zeros((28, 28), dtype=np.uint8)
                return self._build_invalid_response(
                    f"Unable to read image stream: {str(e)}",
                    dummy_vis,
                    confidence_threshold
                )

        # 2. Preprocess using Stage 3 verified pipeline
        processed_batch, is_valid, prep_msg, intermediate_28x28 = preprocess_user_image(
            image_input,
            target_size=(28, 28),
            normalize=True
        )

        # 3. Check image validity (blank / low-contrast / empty defense)
        if not is_valid:
            return self._build_invalid_response(prep_msg, intermediate_28x28, confidence_threshold)

        # 4. Model Forward Pass (inference mode: training=False, no augmentation)
        # Using model(x, training=False) is faster than model.predict for single items
        raw_output = self.model(processed_batch, training=False).numpy()[0]
        probs = np.clip(raw_output, 0.0, 1.0)
        # Re-normalize just in case of float rounding
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum

        # 5. Extract Top predictions
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        top_char = CLASS_MAPPING[top_idx]

        # Top-3 predictions (sorted descending)
        top_3_indices = np.argsort(probs)[::-1][:3]
        top_3_list = [
            {
                "rank": rank + 1,
                "character": CLASS_MAPPING[idx],
                "probability": float(probs[idx]),
                "percentage": f"{float(probs[idx]) * 100:.2f}%"
            }
            for rank, idx in enumerate(top_3_indices)
        ]

        # 6. Confidence Threshold Evaluation
        if top_conf >= confidence_threshold:
            status = "success"
            prediction = top_char
            message = f"Character identified as '{top_char}' with {top_conf * 100:.2f}% confidence."
        else:
            status = "uncertain"
            prediction = "Uncertain"
            message = (
                f"The model is not confident enough (confidence: {top_conf * 100:.2f}%, "
                f"threshold: {confidence_threshold * 100:.1f}%). "
                f"Please upload a clearer handwritten character."
            )

        return {
            "is_valid": True,
            "status": status,
            "prediction": prediction,
            "raw_prediction": top_char,
            "confidence": top_conf,
            "confidence_percent": round(top_conf * 100, 2),
            "top_3": top_3_list,
            "all_probabilities": [float(p) for p in probs],
            "threshold": confidence_threshold,
            "message": message,
            "preprocessed_image_28x28": intermediate_28x28
        }

    def _build_invalid_response(self, reason: str, vis_img: np.ndarray, threshold: float) -> dict[str, Any]:
        """Generate a structured, safe response for blank/invalid inputs."""
        empty_top_3 = [
            {"rank": i + 1, "character": "-", "probability": 0.0, "percentage": "0.00%"}
            for i in range(3)
        ]
        return {
            "is_valid": False,
            "status": "invalid",
            "prediction": "Invalid Input",
            "raw_prediction": None,
            "confidence": 0.0,
            "confidence_percent": 0.0,
            "top_3": empty_top_3,
            "all_probabilities": [0.0] * NUM_CLASSES,
            "threshold": threshold,
            "message": f"Invalid input: {reason}",
            "preprocessed_image_28x28": vis_img
        }

# Convenience function for one-off predictions
def predict_character(
    image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_path: Optional[str] = None
) -> dict[str, Any]:
    """Convenience functional interface that uses the singleton predictor."""
    predictor = CharacterPredictor.get_instance(model_path)
    return predictor.predict(image_input, confidence_threshold=confidence_threshold)
