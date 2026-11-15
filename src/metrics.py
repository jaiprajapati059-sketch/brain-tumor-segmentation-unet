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

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    tf = None
    _HAS_TF = False


def _register_metric(fn):
    """Safely register metric with Keras if TensorFlow is available."""
    if _HAS_TF and hasattr(tf, "keras") and hasattr(tf.keras.utils, "register_keras_serializable"):
        return tf.keras.utils.register_keras_serializable(package="MedicalMetrics")(fn)
    return fn


@_register_metric
def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    """Computes the Sørensen-Dice Similarity Coefficient for tensors."""
    if not _HAS_TF:
        raise ImportError("TensorFlow is required for tensor metric calculations. Please run `pip install -r requirements.txt`.")
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    cardinality = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice


@_register_metric
def iou_metric(y_true, y_pred, smooth: float = 1e-6):
    """Computes the Intersection over Union (IoU / Jaccard Index) for tensors."""
    if not _HAS_TF:
        raise ImportError("TensorFlow is required for tensor metric calculations.")
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


@_register_metric
def precision_metric(y_true, y_pred, smooth: float = 1e-6):
    """Computes Precision (Positive Predictive Value) for segmentation masks."""
    if not _HAS_TF:
        raise ImportError("TensorFlow is required for tensor metric calculations.")
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    predicted_positives = tf.reduce_sum(y_pred_f)

    return (true_positives + smooth) / (predicted_positives + smooth)


@_register_metric
def recall_metric(y_true, y_pred, smooth: float = 1e-6):
    """Computes Recall (Sensitivity) for segmentation masks."""
    if not _HAS_TF:
        raise ImportError("TensorFlow is required for tensor metric calculations.")
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

