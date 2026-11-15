"""
Evaluation & Inference Script for Brain Tumor Semantic Segmentation
Brain Tumor Semantic Segmentation - LGG MRI Dataset (Buda et al.)

Evaluates the trained U-Net on the reserved, patient-isolated Test Set:
  1. Computes quantitative metrics: Overall Dice Score, IoU, Positive-Slice Dice, Specificity.
  2. Generates visual inference grids (Raw MRI vs Ground Truth vs Prediction).
  3. Saves metrics report to JSON.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.config import Config
from src.dataset import (
    load_and_preprocess_sample,
    parse_lgg_dataset,
    split_dataset_by_patient,
    LGGSequence
)
from src.losses import CombinedBCEDiceLoss, dice_loss, bce_dice_loss
from src.metrics import (
    dice_coef,
    iou_metric,
    precision_metric,
    recall_metric,
    numpy_dice_coefficient,
    numpy_iou
)
from src.utils import set_seed, visualize_test_predictions


def load_best_model(model_path: str) -> tf.keras.Model:
    """Loads a saved Keras model along with registered custom loss and metric functions."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{model_path}'. "
            "Please run `python src/train.py` first to train the network."
        )

    custom_objects = {
        "CombinedBCEDiceLoss": CombinedBCEDiceLoss,
        "bce_dice_loss": bce_dice_loss,
        "dice_loss": dice_loss,
        "dice_coef": dice_coef,
        "iou_metric": iou_metric,
        "precision_metric": precision_metric,
        "recall_metric": recall_metric,
    }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model


def evaluate_test_set(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, Any]:
    """Computes comprehensive quantitative test metrics on the patient-isolated test set.

    Calculates:
      - Overall Average Dice Similarity Coefficient (DSC)
      - Overall Average Intersection over Union (IoU)
      - Positive-Slice Dice (performance specifically on slices with visible tumors)
      - Negative-Slice False Positive Rate (specificity on clean brain slices)
      - Standard deviations and confidence metrics

    Args:
        model: Trained U-Net model.
        test_df: Metadata DataFrame of test slices.
        threshold: Sigmoid binarization threshold (default: 0.5).
        verbose: Whether to print summary table.

    Returns:
        results_dict: Dictionary containing metric values.
    """
    dice_scores = []
    iou_scores = []
    pos_dice_scores = []
    neg_clean_flags = []

    print(f"\nEvaluating model across {len(test_df)} reserved test slices...")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test Set Evaluation"):
        img, gt_mask = load_and_preprocess_sample(
            row["image_path"],
            row["mask_path"],
            target_size=(256, 256),
            augment=False
        )

        input_batch = np.expand_dims(img, axis=0)
        pred_prob = model.predict(input_batch, verbose=0)[0]
        pred_mask = (pred_prob > threshold).astype(np.float32)

        has_tumor_gt = np.any(gt_mask > 0)
        slice_dice = numpy_dice_coefficient(gt_mask, pred_mask, threshold=threshold)
        slice_iou = numpy_iou(gt_mask, pred_mask, threshold=threshold)

        dice_scores.append(slice_dice)
        iou_scores.append(slice_iou)

        if has_tumor_gt:
            pos_dice_scores.append(slice_dice)
        else:
            # Check if prediction is properly clean (zero false positives)
            pred_tumor_pixels = np.sum(pred_mask)
            neg_clean_flags.append(pred_tumor_pixels == 0)

    mean_dice = float(np.mean(dice_scores))
    std_dice = float(np.std(dice_scores))
    mean_iou = float(np.mean(iou_scores))
    std_iou = float(np.std(iou_scores))

    mean_pos_dice = float(np.mean(pos_dice_scores)) if pos_dice_scores else 0.0
    neg_specificity = float(np.mean(neg_clean_flags)) if neg_clean_flags else 1.0

    results = {
        "num_test_patients": int(test_df["patient_id"].nunique()),
        "num_test_slices": int(len(test_df)),
        "num_tumor_positive_slices": int(len(pos_dice_scores)),
        "num_tumor_negative_slices": int(len(neg_clean_flags)),
        "test_mean_dice": round(mean_dice, 4),
        "test_std_dice": round(std_dice, 4),
        "test_mean_iou": round(mean_iou, 4),
        "test_std_iou": round(std_iou, 4),
        "positive_slice_mean_dice": round(mean_pos_dice, 4),
        "negative_slice_specificity": round(neg_specificity, 4),
    }

    if verbose:
        print("\n" + "=" * 65)
        print("QUANTITATIVE TEST SET BENCHMARK RESULTS")
        print("=" * 65)
        print(f"Isolated Test Patients     : {results['num_test_patients']}")
        print(f"Total Test MRI Slices       : {results['num_test_slices']}")
        print(f"  - Tumor-Positive Slices   : {results['num_tumor_positive_slices']}")
        print(f"  - Tumor-Free Slices       : {results['num_tumor_negative_slices']}")
        print("-" * 65)
        print(f"Overall Dice Score (DSC)    : {results['test_mean_dice']:.4f} ± {results['test_std_dice']:.4f}")
        print(f"Overall IoU (Jaccard Index) : {results['test_mean_iou']:.4f} ± {results['test_std_iou']:.4f}")
        print(f"Tumor-Positive Slices Dice  : {results['positive_slice_mean_dice']:.4f}")
        print(f"Healthy Slice Specificity   : {results['negative_slice_specificity'] * 100:.2f}%")
        print("=" * 65 + "\n")

    return results


def run_evaluation(config: Config) -> None:
    """Full evaluation and visualization pipeline."""
    config.ensure_directories()
    set_seed(config.random_seed)

    # 1. Load Model
    print(f"Loading trained weights from: {config.model_save_path}")
    model = load_best_model(config.model_save_path)

    # 2. Load or Reconstruct Test Set
    test_csv_path = os.path.join(config.output_dir, "test_split.csv")
    if os.path.exists(test_csv_path):
        print(f"Loading cached test partition from '{test_csv_path}'...")
        test_df = pd.read_csv(test_csv_path)
    else:
        print("Cached test partition not found. Parsing dataset to perform deterministic split...")
        full_df = parse_lgg_dataset(config.data_dir)
        _, _, test_df = split_dataset_by_patient(
            full_df,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            random_seed=config.random_seed
        )

    # 3. Quantitative Test Metrics
    metrics = evaluate_test_set(model, test_df, threshold=config.prediction_threshold)

    # Save metrics to JSON
    with open(config.eval_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved quantitative test metrics to: {config.eval_save_path}")

    # 4. Generate Visual Inference Grids
    vis_path = os.path.join(config.output_dir, "test_visual_inference.png")
    print(f"\nGenerating visual inference panels for {config.num_visual_samples} test slices...")
    visualize_test_predictions(
        model=model,
        test_df=test_df,
        num_samples=config.num_visual_samples,
        threshold=config.prediction_threshold,
        save_path=vis_path,
        random_seed=config.random_seed
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate U-Net on LGG MRI Test Set")
    parser.add_argument("--model_path", type=str, default=None, help="Custom path to model .keras file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of visual inference samples")
    parser.add_argument("--config", type=str, default=None, help="Optional custom YAML config")

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()

    if args.model_path:
        cfg.best_model_name = os.path.basename(args.model_path)
        cfg.checkpoint_dir = os.path.dirname(args.model_path)
    cfg.output_dir = args.output_dir
    cfg.num_visual_samples = args.num_samples

    run_evaluation(cfg)


if __name__ == "__main__":
    main()

