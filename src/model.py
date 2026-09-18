"""Convolutional Neural Network (CNN) architecture for 26-class handwritten character recognition."""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Single authoritative class mapping: 0 -> 'A', ..., 25 -> 'Z'
CLASS_MAPPING = {i: chr(ord('A') + i) for i in range(26)}
NUM_CLASSES = 26
INPUT_SHAPE = (28, 28, 1)

def build_character_cnn(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    include_augmentation: bool = True,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Build a clean, beginner-friendly yet robust CNN for 26-class character recognition.
    
    Architecture:
      Input (28, 28, 1)
      [Optional Augmentation: active only during training]
      -> Conv2D (32 filters, 3x3, padding='same', ReLU)
      -> BatchNormalization
      -> MaxPooling2D (2, 2)
      -> Conv2D (64 filters, 3x3, padding='same', ReLU)
      -> BatchNormalization
      -> MaxPooling2D (2, 2)
      -> Conv2D (64 filters, 3x3, padding='same', ReLU)
      -> MaxPooling2D (2, 2)
      -> Flatten
      -> Dense (128 units, ReLU)
      -> Dropout (0.35)
      -> Dense (26 units, Softmax)
    
    Args:
        input_shape: (28, 28, 1)
        num_classes: 26
        include_augmentation: If True, prepends Keras data augmentation layers
                             which automatically activate only during model.fit().
        learning_rate: Initial Adam learning rate.
        
    Returns:
        Compiled Keras Model.
    """
    inputs = keras.Input(shape=input_shape, name="character_image_input")
    
    x = inputs
    if include_augmentation:
        # Augmentation is active during model.fit (training=True) and bypassed in model.predict (training=False)
        x = layers.RandomRotation(factor=0.06, fill_mode="constant", fill_value=0.0, name="aug_rotation")(x)
        x = layers.RandomTranslation(height_factor=0.06, width_factor=0.06, fill_mode="constant", fill_value=0.0, name="aug_translation")(x)
        x = layers.RandomZoom(height_factor=(-0.06, 0.06), width_factor=(-0.06, 0.06), fill_mode="constant", fill_value=0.0, name="aug_zoom")(x)

    # Block 1
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Dense classifier head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.35, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="handwritten_character_cnn")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def load_trained_model(model_path: str = None) -> keras.Model:
    """
    Safely load the trained .keras model.
    """
    if model_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "models", "handwritten_character_model.keras")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Run training/train.py first.")
    
    return keras.models.load_model(model_path)
