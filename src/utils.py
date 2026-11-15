"""
Visualization & Utility Functions
Brain Tumor Semantic Segmentation - LGG MRI Dataset (Buda et al.)

Provides publication-quality plotting, mask overlay generation, reproducible seeding,
and inference visualizer tools.
"""

import os
import random
from typing import Optional, List, Tuple, Union, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import tensorflow as tf

from src.dataset import load_and_preprocess_sample
from src.metrics import numpy_dice_coefficient, numpy_iou


def set_seed(seed: int = 42) -> None:
    """Enforces deterministic reproducibility across all runtime environments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.40,
    draw_contour: bool = True
) -> np.ndarray:
    """Blends a binary segmentation mask onto a 3-channel MRI slice.

    Args:
        image: RGB float array in range [0.0, 1.0] of shape (H, W, 3).
        mask: Binary mask array (H, W) or (H, W, 1) with values in {0.0, 1.0}.
        color: RGB color tuple for the overlay (default: Red (1.0, 0.0, 0.0)).
        alpha: Blending transparency factor.
        draw_contour: Whether to draw a crisp boundary outline around the tumor.

    Returns:
        blended: RGB float array of shape (H, W, 3) ready for display or saving.
    """
    img_rgb = image.copy()
    if img_rgb.max() > 1.0:
        img_rgb = img_rgb / 255.0

    mask_2d = mask.squeeze()
    binary_mask = (mask_2d > 0.5).astype(np.uint8)

    # Create solid color mask
    color_mask = np.zeros_like(img_rgb)
    for c in range(3):
        color_mask[:, :, c] = binary_mask * color[c]

    # Alpha blend where mask is active
    blended = np.where(
        binary_mask[:, :, None] == 1,
        (1.0 - alpha) * img_rgb + alpha * color_mask,
        img_rgb
    )

    # Optional contour outline for clinical boundary delineation
    if draw_contour and np.any(binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blended_uint8 = (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)
        contour_color_bgr = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        cv2.drawContours(blended_uint8, contours, -1, contour_color_bgr, thickness=1)
        blended = blended_uint8.astype(np.float32) / 255.0

    return np.clip(blended, 0.0, 1.0)


def plot_training_history(
    history: Union[tf.keras.callbacks.History, Dict[str, list]],
    save_path: Optional[str] = "outputs/training_curves.png"
) -> None:
    """Generates publication-quality training and validation loss/metric curves.

    Args:
        history: Keras History object or history dictionary.
        save_path: File path to save the generated plot.
    """
    hist = history.history if hasattr(history, "history") else history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#ffffff")

    # 1. Loss Subplot
    axes[0].plot(epochs, hist["loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2, markersize=4)
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], "s--", color="#ff7f0e", label="Val Loss", linewidth=2, markersize=4)
    axes[0].set_title("Training vs. Validation Loss (BCE + Dice)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Dice Coefficient Subplot
    axes[1].plot(epochs, hist["dice_coef"], "o-", color="#2ca02c", label="Train Dice", linewidth=2, markersize=4)
    if "val_dice_coef" in hist:
        axes[1].plot(epochs, hist["val_dice_coef"], "s--", color="#d62728", label="Val Dice", linewidth=2, markersize=4)
    axes[1].set_title("Dice Similarity Coefficient (DSC)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Dice Score", fontsize=11)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. IoU (Jaccard Index) Subplot
    iou_key = "iou_metric" if "iou_metric" in hist else "iou"
    val_iou_key = f"val_{iou_key}"
    if iou_key in hist:
        axes[2].plot(epochs, hist[iou_key], "o-", color="#9467bd", label="Train IoU", linewidth=2, markersize=4)
        if val_iou_key in hist:
            axes[2].plot(epochs, hist[val_iou_key], "s--", color="#8c564b", label="Val IoU", linewidth=2, markersize=4)
        axes[2].set_title("Intersection over Union (IoU)", fontsize=13, fontweight="bold")
        axes[2].set_xlabel("Epoch", fontsize=11)
        axes[2].set_ylabel("IoU Score", fontsize=11)
        axes[2].legend(frameon=True)
        axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training history curves to: {save_path}")
    plt.close()


def visualize_test_predictions(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    num_samples: int = 5,
    threshold: float = 0.5,
    save_path: Optional[str] = "outputs/test_visual_inference.png",
    random_seed: int = 42
) -> None:
    """Randomly samples slices from the reserved Test Set and generates a 1x3 grid:
       - Pane 1: Original raw MRI
       - Pane 2: Ground Truth (Original MRI overlaid with dataset ground truth mask)
       - Pane 3: Prediction (Original MRI overlaid with model's predicted mask)

    Intelligently balances the selection by prioritizing tumor-positive slices
    so the visualization demonstrates true lesion delineation ability.

    Args:
        model: Trained U-Net Keras model.
        test_df: Test DataFrame with 'image_path' and 'mask_path'.
        num_samples: Number of patient slices to visualize.
        threshold: Sigmoid probability threshold for binary classification.
        save_path: Output file path for the visualization image.
        random_seed: Seed for reproducible sample selection.
    """
    np.random.seed(random_seed)

    # Identify slices with tumors (mask non-zero) for meaningful visualization
    has_tumor_list = []
    for _, row in test_df.iterrows():
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        has_tumor_list.append(np.any(mask > 0) if mask is not None else False)

    test_df = test_df.copy()
    test_df["has_tumor"] = has_tumor_list

    pos_df = test_df[test_df["has_tumor"]].reset_index(drop=True)
    neg_df = test_df[~test_df["has_tumor"]].reset_index(drop=True)

    # Select mostly positive slices + at least one negative slice if available
    selected_rows = []
    if len(pos_df) > 0:
        pos_sample_count = min(len(pos_df), num_samples - 1 if len(neg_df) > 0 else num_samples)
        selected_rows.extend(pos_df.sample(n=pos_sample_count, random_state=random_seed).to_dict("records"))
    if len(neg_df) > 0 and len(selected_rows) < num_samples:
        neg_sample_count = min(len(neg_df), num_samples - len(selected_rows))
        selected_rows.extend(neg_df.sample(n=neg_sample_count, random_state=random_seed).to_dict("records"))

    # Fallback if filtered sets are smaller than num_samples
    if len(selected_rows) < num_samples:
        selected_rows = test_df.sample(n=min(len(test_df), num_samples), random_state=random_seed).to_dict("records")

    total_samples = len(selected_rows)
    fig, axes = plt.subplots(total_samples, 3, figsize=(15, 4.5 * total_samples))
    if total_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.patch.set_facecolor("#1e1e1e")

    for i, row in enumerate(selected_rows):
        img, gt_mask = load_and_preprocess_sample(
            row["image_path"],
            row["mask_path"],
            target_size=(256, 256),
            augment=False
        )

        # Model inference
        input_tensor = np.expand_dims(img, axis=0)
        pred_prob = model.predict(input_tensor, verbose=0)[0]
        pred_mask = (pred_prob > threshold).astype(np.float32)

        # Quantitative metrics for this slice
        slice_dice = numpy_dice_coefficient(gt_mask, pred_mask, threshold=threshold)
        slice_iou = numpy_iou(gt_mask, pred_mask, threshold=threshold)

        # Overlays: Ground truth in Emerald Green, Prediction in Crimson Red
        gt_overlay = create_mask_overlay(img, gt_mask, color=(0.1, 0.9, 0.3), alpha=0.45)
        pred_overlay = create_mask_overlay(img, pred_mask, color=(1.0, 0.2, 0.2), alpha=0.45)

        # Pane 1: Raw MRI
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Patient: {row['patient_id']}\nRaw MRI (FLAIR)", color="white", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # Pane 2: Ground Truth Overlay
        axes[i, 1].imshow(gt_overlay)
        gt_label = "Positive Tumor" if np.any(gt_mask > 0) else "Negative (Healthy)"
        axes[i, 1].set_title(f"Ground Truth\n({gt_label})", color="#2ecc71", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")

        # Pane 3: Prediction Overlay
        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title(
            f"U-Net Prediction\nDice: {slice_dice:.3f} | IoU: {slice_iou:.3f}",
            color="#e74c3c" if slice_dice < 0.70 else "#2ecc71",
            fontsize=11,
            fontweight="bold"
        )
        axes[i, 2].axis("off")

    # Add shared legend at bottom
    gt_patch = mpatches.Patch(color=(0.1, 0.9, 0.3), label="Ground Truth Annotation")
    pred_patch = mpatches.Patch(color=(1.0, 0.2, 0.2), label="U-Net Predicted Segment")
    fig.legend(
        handles=[gt_patch, pred_patch],
        loc="lower center",
        ncol=2,
        frameon=True,
        facecolor="#2d2d2d",
        edgecolor="none",
        labelcolor="white",
        fontsize=12,
        bbox_to_anchor=(0.5, 0.01)
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved test inference visualizations to: {save_path}")
    plt.close()

