"""
Training Pipeline for Brain Tumor Semantic Segmentation
Brain Tumor Semantic Segmentation - LGG MRI Dataset (Buda et al.)

Features:
  - Patient-level splitting ensuring 0% data leakage
  - Data augmentation for medical imaging
  - U-Net compilation with Combined BCE + Dice Loss
  - EarlyStopping & ModelCheckpoint callbacks monitoring val_dice_coef
  - History logging to CSV and automated curve generation
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import tensorflow as tf

from src.config import Config
from src.dataset import (
    parse_lgg_dataset,
    split_dataset_by_patient,
    LGGSequence,
    create_dummy_dataset,
)
from src.model import build_unet
from src.losses import CombinedBCEDiceLoss
from src.metrics import dice_coef, iou_metric, precision_metric, recall_metric
from src.utils import set_seed, plot_training_history


def train_pipeline(config: Config, use_synthetic: bool = False) -> tf.keras.callbacks.History:
    """Executes the complete end-to-end training pipeline.

    Args:
        config: Central configuration object.
        use_synthetic: If True, generates a synthetic dataset for quick testing.

    Returns:
        history: Keras training history object.
    """
    config.ensure_directories()
    set_seed(config.random_seed)

    print("\n" + "=" * 70)
    print("STARTING BRAIN TUMOR SEGMENTATION TRAINING PIPELINE")
    print(f"Dataset Target: {config.data_dir}")
    print(f"Batch Size: {config.batch_size} | Epochs: {config.epochs} | Init LR: {config.initial_learning_rate}")
    print("=" * 70 + "\n")

    # 1. Dataset Discovery / Synthesis
    data_path = Path(config.data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        if use_synthetic:
            print(f"Dataset directory '{config.data_dir}' not found. Generating synthetic LGG MRI dataset...")
            create_dummy_dataset(output_dir=config.data_dir, num_patients=8, slices_per_patient=14)
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{config.data_dir}'.\n"
                f"Please download the Kaggle LGG dataset or re-run with '--synthetic' to test with mock MRIs."
            )

    # 2. Parse Filepaths & Extract Patient IDs
    print("[1/5] Parsing dataset directory and extracting Patient IDs...")
    full_df = parse_lgg_dataset(config.data_dir)
    print(f"Discovered {len(full_df)} total MRI slices across {full_df['patient_id'].nunique()} unique patients.")

    # 3. Patient-Level Splitting (CRITICAL)
    print("\n[2/5] Performing Patient-Level Split (Train 70% / Val 15% / Test 15%)...")
    train_df, val_df, test_df = split_dataset_by_patient(
        full_df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
        verbose=True
    )

    # Persist split metadata to outputs directory for deterministic evaluation
    train_df.to_csv(os.path.join(config.output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(config.output_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(config.output_dir, "test_split.csv"), index=False)
    print(f"Saved partition metadata to '{config.output_dir}/'")

    # 4. Initialize Data Generators
    print("\n[3/5] Instantiating batch generators with data augmentation...")
    train_gen = LGGSequence(
        df=train_df,
        batch_size=config.batch_size,
        target_size=(config.image_height, config.image_width),
        augment=True,
        shuffle=True
    )
    val_gen = LGGSequence(
        df=val_df,
        batch_size=config.batch_size,
        target_size=(config.image_height, config.image_width),
        augment=False,
        shuffle=False
    )

    # 5. Build & Compile U-Net
    print("\n[4/5] Building U-Net from scratch with skip connections...")
    model = build_unet(
        input_shape=(config.image_height, config.image_width, config.num_channels),
        num_classes=config.num_classes,
        base_filters=32,
        dropout_rate=0.2
    )

    loss_fn = CombinedBCEDiceLoss(
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        smooth=config.smooth_factor
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.initial_learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            dice_coef,
            iou_metric,
            precision_metric,
            recall_metric
        ]
    )

    # 6. Configure Callbacks
    callbacks = [
        # Save best weights based on validation Dice Similarity Coefficient
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.model_save_path,
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        # Prevent overfitting if validation Dice stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=config.patience_early_stopping,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss/dice plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coef",
            mode="max",
            factor=config.reduce_lr_factor,
            patience=config.patience_reduce_lr,
            min_lr=config.min_learning_rate,
            verbose=1
        ),
        # Log epoch metrics to CSV
        tf.keras.callbacks.CSVLogger(
            filename=config.log_save_path,
            separator=",",
            append=False
        )
    ]

    # 7. Execute Training Loop
    print("\n[5/5] Training model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Plot and Save Learning Curves
    curves_path = os.path.join(config.output_dir, "training_curves.png")
    plot_training_history(history, save_path=curves_path)

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print(f"Best model weights saved to: {config.model_save_path}")
    print(f"Training metrics log saved to: {config.log_save_path}")
    print(f"Learning curves plot saved to: {curves_path}")
    print("Run `python src/evaluate.py` to evaluate performance on the reserved Test Set.")
    print("=" * 70 + "\n")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for Brain Tumor Segmentation (LGG MRI)")
    parser.add_argument("--data_dir", type=str, default="data/kaggle_3m", help="Path to raw dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial Adam learning rate")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dataset for testing if dataset is missing")
    parser.add_argument("--config", type=str, default=None, help="Optional path to custom YAML config")

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()

    cfg.data_dir = args.data_dir
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.initial_learning_rate = args.lr

    train_pipeline(cfg, use_synthetic=args.synthetic)


if __name__ == "__main__":
    main()

