"""Verification script for Stage 1 environment setup."""
import sys

def test_environment():
    import tensorflow as tf
    import keras
    import numpy as np
    import cv2
    import PIL
    import matplotlib
    import sklearn
    import streamlit

    print("=== ENVIRONMENT VERIFICATION SUCCESSFUL ===")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Pillow Version: {PIL.__version__}")
    print(f"Matplotlib Version: {matplotlib.__version__}")
    print(f"Scikit-Learn Version: {sklearn.__version__}")
    print(f"Streamlit Version: {streamlit.__version__}")
    print("==========================================")

if __name__ == "__main__":
    test_environment()
