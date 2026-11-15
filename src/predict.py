"""
Standalone Single-Image Inference Script for Brain Tumor Segmentation
Brain Tumor Semantic Segmentation - LGG MRI Dataset (Buda et al.)

Usage:
  python src/predict.py --image path/to/mri_slice.tif --model checkpoints/best_unet_lgg.keras --output outputs/prediction.png
"""

import os
import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from src.evaluate import load_best_model
from src.utils import create_mask_overlay
from src.metrics import numpy_dice_coefficient, numpy_iou


def predict_single_slice(
    model_path: str,
    image_path: str,
    mask_path: str = None,
    threshold: float = 0.5,
    output_path: str = "outputs/single_prediction.png"
):
    """Executes segmentation on a single MRI slice and creates an overlay comparison.

    Args:
        model_path: Path to saved .keras checkpoint.
        image_path: Path to raw input MRI slice.
        mask_path: Optional path to ground truth mask for benchmarking.
        threshold: Binarization probability threshold.
        output_path: Destination path for the visual comparison image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # 1. Load trained model
    print(f"Loading model from: {model_path}")
    model = load_best_model(model_path)

    # 2. Preprocess MRI slice
    raw_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if raw_img is None:
        raise ValueError(f"Could not read image file at {image_path}")
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = raw_img.shape

    resized_img = cv2.resize(raw_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    norm_img = resized_img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(norm_img, axis=0)

    # 3. Model Forward Pass
    print("Running neural network forward pass...")
    pred_prob = model.predict(input_tensor, verbose=0)[0]
    pred_mask_256 = (pred_prob > threshold).astype(np.float32)

    # Resize prediction back to original dimensions
    pred_mask_orig = cv2.resize(pred_mask_256, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    pred_mask_orig = np.expand_dims(pred_mask_orig, axis=-1)

    has_gt = mask_path and os.path.exists(mask_path)
    gt_mask_orig = None
    slice_dice = None
    slice_iou = None

    if has_gt:
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask_orig = (raw_mask > 127).astype(np.float32)
        gt_mask_orig = np.expand_dims(gt_mask_orig, axis=-1)
        slice_dice = numpy_dice_coefficient(gt_mask_orig, pred_mask_orig, threshold=0.5)
        slice_iou = numpy_iou(gt_mask_orig, pred_mask_orig, threshold=0.5)

    # 4. Generate Overlays
    pred_overlay = create_mask_overlay(raw_img / 255.0, pred_mask_orig, color=(1.0, 0.2, 0.2), alpha=0.45)

    # 5. Render Comparison Figure
    num_cols = 3 if has_gt else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
    fig.patch.set_facecolor("#1e1e1e")

    # Pane 1: Raw MRI
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Input MRI Slice\n({Path(image_path).name})", color="white", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    col_idx = 1
    if has_gt:
        gt_overlay = create_mask_overlay(raw_img / 255.0, gt_mask_orig, color=(0.1, 0.9, 0.3), alpha=0.45)
        axes[col_idx].imshow(gt_overlay)
        axes[col_idx].set_title("Ground Truth Mask\n(Radiologist Annotation)", color="#2ecc71", fontsize=11, fontweight="bold")
        axes[col_idx].axis("off")
        col_idx += 1

    # Predicted Overlay Pane
    axes[col_idx].imshow(pred_overlay)
    pred_title = "U-Net Predicted Tumor"
    if slice_dice is not None:
        pred_title += f"\nDice: {slice_dice:.3f} | IoU: {slice_iou:.3f}"
    axes[col_idx].set_title(pred_title, color="#e74c3c", fontsize=11, fontweight="bold")
    axes[col_idx].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"Prediction successful! Visual output saved to: {output_path}")
    if slice_dice is not None:
        print(f"Quantitative Score -> Dice: {slice_dice:.4f} | IoU: {slice_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Predict Brain Tumor Segmentation on a Single MRI Slice")
    parser.add_argument("--image", type=str, required=True, help="Path to input MRI slice (.tif or image)")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to ground truth mask")
    parser.add_argument("--model", type=str, default="checkpoints/best_unet_lgg.keras", help="Path to model weights")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability binarization threshold")
    parser.add_argument("--output", type=str, default="outputs/single_prediction.png", help="Path to save result")

    args = parser.parse_args()
    predict_single_slice(
        model_path=args.model,
        image_path=args.image,
        mask_path=args.mask,
        threshold=args.threshold,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
