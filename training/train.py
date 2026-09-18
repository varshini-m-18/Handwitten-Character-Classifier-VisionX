"""Training script for handwritten character classifier CNN."""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import load_emnist_letters
from src.preprocessing import preprocess_dataset_images
from src.model import build_character_cnn, NUM_CLASSES, INPUT_SHAPE

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for Handwritten Character Classification (A-Z)")
    parser.add_argument("--epochs", type=int, default=12, help="Maximum number of training epochs (default: 12)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial Adam learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (default: 3)")
    parser.add_argument("--val_split", type=float, default=0.10, help="Fraction of train data for validation (default: 0.10)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()

def train():
    args = parse_args()
    print("=" * 60)
    print("HANDWRITTEN CHARACTER CLASSIFIER (A-Z) — TRAINING")
    print("=" * 60)
    print(f"Configuration: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.learning_rate}, Patience={args.patience}")

    models_dir = os.path.join(PROJECT_ROOT, "models")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "handwritten_character_model.keras")

    # 1. Load dataset
    print("\n[1/6] Loading EMNIST Letters dataset...")
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw), label_map = load_emnist_letters(fix_zero_indexed=True)
    print(f"Loaded {len(X_train_raw):,} raw training samples and {len(X_test_raw):,} raw testing samples.")

    # 2. Stratified train/validation split
    print(f"\n[2/6] Splitting training data into Train and Validation ({int((1-args.val_split)*100)}% / {int(args.val_split*100)}%)...")
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_raw,
        y_train_raw,
        test_size=args.val_split,
        random_state=args.random_state,
        stratify=y_train_raw
    )
    print(f"  Training samples:   {len(X_train_split):,}")
    print(f"  Validation samples: {len(X_val_split):,}")
    print(f"  Untouched test set: {len(X_test_raw):,}")

    # 3. Preprocess images
    print("\n[3/6] Preprocessing image tensors to float32 [0.0, 1.0]...")
    X_train = preprocess_dataset_images(X_train_split)
    X_val = preprocess_dataset_images(X_val_split)
    X_test = preprocess_dataset_images(X_test_raw)
    y_train = y_train_split
    y_val = y_val_split
    y_test = y_test_raw

    # Sanity checks
    assert X_train.shape[1:] == INPUT_SHAPE, f"Train shape error: {X_train.shape}"
    assert X_val.shape[1:] == INPUT_SHAPE, f"Val shape error: {X_val.shape}"
    assert X_test.shape[1:] == INPUT_SHAPE, f"Test shape error: {X_test.shape}"
    assert 0 <= y_train.min() and y_train.max() < NUM_CLASSES, "Label error in training set"
    assert 0 <= y_val.min() and y_val.max() < NUM_CLASSES, "Label error in validation set"

    # 4. Build CNN Model
    print("\n[4/6] Building CNN architecture with training augmentation...")
    model = build_character_cnn(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        include_augmentation=True,
        learning_rate=args.learning_rate
    )
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 5. Model Training
    print("\n[5/6] Starting CNN training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    epochs_completed = len(history.history["loss"])
    best_val_acc = float(max(history.history["val_accuracy"]))
    final_train_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Epochs completed:       {epochs_completed}/{args.epochs}")
    print(f"Best Validation Acc:    {best_val_acc * 100:.2f}%")
    print(f"Final Training Acc:     {final_train_acc * 100:.2f}%")
    print(f"Final Validation Acc:   {final_val_acc * 100:.2f}%")
    print(f"Model saved to:         {model_save_path}")

    # 6. Save Training History & Plot
    print("\n[6/6] Saving training history and generating plots...")
    hist_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    hist_json_path = os.path.join(results_dir, "training_history.json")
    with open(hist_json_path, "w") as f:
        json.dump(hist_dict, f, indent=2)

    # Plot Accuracy & Loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs_range = range(1, epochs_completed + 1)

    # Accuracy plot
    ax1.plot(epochs_range, [acc * 100 for acc in history.history["accuracy"]], "o-", label="Training Accuracy", color="#1f77b4", linewidth=2)
    ax1.plot(epochs_range, [acc * 100 for acc in history.history["val_accuracy"]], "s-", label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    ax1.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss plot
    ax2.plot(epochs_range, history.history["loss"], "o-", label="Training Loss", color="#1f77b4", linewidth=2)
    ax2.plot(epochs_range, history.history["val_loss"], "s-", label="Validation Loss", color="#ff7f0e", linewidth=2)
    ax2.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Loss (Sparse Categorical Crossentropy)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.suptitle("CNN Model Training Performance — 26-Class Character Recognition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training plot to: {plot_path}")

    # Evaluate best model on test set
    print("\nEvaluating saved best model on untouched test set (20,800 images)...")
    best_model = keras.models.load_model(model_save_path)
    test_loss, test_acc = best_model.evaluate(X_test, y_test, batch_size=args.batch_size, verbose=1)
    print(f"\n>>> TEST SET RESULTS: Loss = {test_loss:.4f}, Accuracy = {test_acc * 100:.2f}% <<<")

    return {
        "epochs_completed": epochs_completed,
        "best_val_acc": best_val_acc,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }

if __name__ == "__main__":
    train()
