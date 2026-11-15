"""
Custom Evaluation Metrics for Medical Semantic Segmentation
Brain Tumor Semantic Segmentation - LGG MRI Dataset (Buda et al.)

Why Standard Accuracy is Invalid for Medical Image Segmentation:
In brain MRI slices, tumor lesions typically occupy less than 1% to 5% of total
slice area (extreme spatial class imbalance). Many slices contain zero tumor pixels.
A naive model predicting pure background (all zeros) achieves >98% pixel accuracy
while having 0% clinical utility. Overlap metrics like Dice Similarity Coefficient (DSC)
and Intersection over Union (IoU / Jaccard Index) evaluate true lesion boundary concordance.
"""

from typing import Union
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="MedicalMetrics")
def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Computes the Sørensen-Dice Similarity Coefficient for tensors.

    Mathematical Formulation:
        Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)

    Args:
        y_true: Ground truth binary tensor (values in {0, 1}).
        y_pred: Predicted probability tensor (values in [0, 1]).
        smooth: Smoothing epsilon preventing division by zero and stabilizing gradients.

    Returns:
        tf.Tensor: Scalar Dice coefficient in [0.0, 1.0].
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    cardinality = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice


@tf.keras.utils.register_keras_serializable(package="MedicalMetrics")
def iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Computes the Intersection over Union (IoU / Jaccard Index) for tensors.

    Mathematical Formulation:
        IoU = |X ∩ Y| / |X ∪ Y| = (|X ∩ Y| + smooth) / (|X| + |Y| - |X ∩ Y| + smooth)

    Args:
        y_true: Ground truth binary tensor.
        y_pred: Predicted probability tensor.
        smooth: Epsilon smoothing constant.

    Returns:
        tf.Tensor: Scalar IoU score in [0.0, 1.0].
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


@tf.keras.utils.register_keras_serializable(package="MedicalMetrics")
def precision_metric(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Computes Precision (Positive Predictive Value) for segmentation masks."""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    predicted_positives = tf.reduce_sum(y_pred_f)

    return (true_positives + smooth) / (predicted_positives + smooth)


@tf.keras.utils.register_keras_serializable(package="MedicalMetrics")
def recall_metric(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Computes Recall (Sensitivity) for segmentation masks."""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    actual_positives = tf.reduce_sum(y_true_f)

    return (true_positives + smooth) / (actual_positives + smooth)


# =============================================================================
# NUMPY EVALUATION FUNCTIONS (FOR TEST SET INFERENCE & BENCHMARKING)
# =============================================================================

def numpy_dice_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """Numpy-based Dice Coefficient calculation on binarized predictions."""
    y_true_bin = (y_true > 0.5).astype(np.float32).flatten()
    y_pred_bin = (y_pred > threshold).astype(np.float32).flatten()

    intersection = np.sum(y_true_bin * y_pred_bin)
    cardinality = np.sum(y_true_bin) + np.sum(y_pred_bin)

    # Edge-case handling: both true and pred are completely empty
    if cardinality == 0:
        return 1.0

    return float((2.0 * intersection + smooth) / (cardinality + smooth))


def numpy_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """Numpy-based IoU calculation on binarized predictions."""
    y_true_bin = (y_true > 0.5).astype(np.float32).flatten()
    y_pred_bin = (y_pred > threshold).astype(np.float32).flatten()

    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin) - intersection

    # Both empty
    if union == 0:
        return 1.0

    return float((intersection + smooth) / (union + smooth))

