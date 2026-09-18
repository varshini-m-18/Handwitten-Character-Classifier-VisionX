"""Comprehensive evaluation script for trained CNN on the untouched EMNIST test set."""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import load_emnist_letters
from src.preprocessing import preprocess_dataset_images
from src.model import load_trained_model, CLASS_MAPPING, NUM_CLASSES

def evaluate(model_path: str = None):
    print("=" * 60)
    print("MODEL EVALUATION ON UNTOUCHED EMNIST TEST SET")
    print("=" * 60)

    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "handwritten_character_model.keras")
    
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load model
    print(f"\n[1/4] Loading model from: {model_path}")
    model = load_trained_model(model_path)
    print("Model loaded successfully.")

    # 2. Load and preprocess untouched test set
    print("\n[2/4] Loading and preprocessing test set...")
    _, (X_test_raw, y_test), label_map = load_emnist_letters(fix_zero_indexed=True)
    X_test = preprocess_dataset_images(X_test_raw)
    print(f"Test samples: {len(X_test):,} images of shape {X_test.shape[1:]}")

    # 3. Model evaluation & inference
    print("\n[3/4] Running test evaluation and predictions...")
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
    
    y_pred_probs = model.predict(X_test, batch_size=256, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Metrics & Reports
    print("\n[4/4] Computing metrics, classification report, and confusion matrix...")
    target_names = [CLASS_MAPPING[i] for i in range(NUM_CLASSES)]
    
    # Classification report
    report_str = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    print("\nClassification Report Summary:")
    print(f"  Accuracy:  {test_acc * 100:.2f}%")
    print(f"  Macro F1:  {report_dict['macro avg']['f1-score'] * 100:.2f}%")
    print(f"  Weighted F1: {report_dict['weighted avg']['f1-score'] * 100:.2f}%")

    report_file_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_file_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("HANDWRITTEN CHARACTER CLASSIFIER — CLASSIFICATION REPORT\n")
        f.write("Dataset: EMNIST Letters (Untouched Test Set, 20,800 images)\n")
        f.write(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_str)
    print(f"Saved classification report to: {report_file_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("EMNIST Letters (26 Classes: A-Z) — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, target_names, fontsize=9)
    plt.yticks(tick_marks, target_names, fontsize=9)
    plt.xlabel("Predicted Character", fontsize=11, fontweight="bold")
    plt.ylabel("True Character", fontsize=11, fontweight="bold")

    # Add text annotations for diagonal and off-diagonal with reasonable contrast
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            if val > 0:
                plt.text(j, i, format(val, "d"), horizontalalignment="center", verticalalignment="center",
                         color=color, fontsize=6.5)

    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # Save summary metrics JSON
    metrics_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "num_test_samples": len(X_test),
        "per_class_accuracy": {
            target_names[i]: float(cm[i, i] / np.sum(cm[i, :])) for i in range(NUM_CLASSES)
        }
    }
    metrics_json_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved evaluation metrics JSON to: {metrics_json_path}")

    return metrics_summary

if __name__ == "__main__":
    evaluate()
