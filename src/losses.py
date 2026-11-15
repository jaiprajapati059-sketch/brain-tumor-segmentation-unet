"""
Custom Loss Functions for Semantic Segmentation with Extreme Class Imbalance
Brain Tumor Semantic Segmentation - LGG MRI Dataset (Buda et al.)

Rationale:
Binary Cross-Entropy (BCE) calculates pixel-wise classification loss, providing
stable gradients during the initial optimization steps. However, due to class
imbalance (99% background pixels), BCE tends to bias predictions towards 0.
Dice Loss directly optimizes the overlap of foreground (tumor) pixels.
Combining BCE + Dice Loss creates a synergistic loss surface:
  - BCE maintains numerical stability and smooth pixel gradients.
  - Dice Loss enforces overlap maximization on tiny lesions.
"""

import tensorflow as tf
from src.metrics import dice_coef


@tf.keras.utils.register_keras_serializable(package="MedicalLosses")
def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Computes Dice Loss: L_dice = 1.0 - Dice_Coefficient."""
    return 1.0 - dice_coef(y_true, y_pred, smooth=smooth)


@tf.keras.utils.register_keras_serializable(package="MedicalLosses")
def bce_dice_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    smooth: float = 1e-6
) -> tf.Tensor:
    """Computes combined Binary Cross-Entropy + Dice Loss.

    Loss = alpha * BCE(y_true, y_pred) + beta * Dice_Loss(y_true, y_pred)
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Average across spatial dimensions to get per-sample BCE
    bce = tf.reduce_mean(bce)

    d_loss = dice_loss(y_true, y_pred, smooth=smooth)

    return (bce_weight * bce) + (dice_weight * d_loss)


@tf.keras.utils.register_keras_serializable(package="MedicalLosses")
class CombinedBCEDiceLoss(tf.keras.losses.Loss):
    """Keras Loss class implementation for BCE + Dice Loss combination."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-6,
        name: str = "bce_dice_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return bce_dice_loss(
            y_true,
            y_pred,
            bce_weight=self.bce_weight,
            dice_weight=self.dice_weight,
            smooth=self.smooth
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "smooth": self.smooth,
        })
        return config


def get_loss(loss_name: str = "bce_dice", bce_weight: float = 1.0, dice_weight: float = 1.0):
    """Loss factory function."""
    if loss_name.lower() in ["bce_dice", "combined"]:
        return CombinedBCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    elif loss_name.lower() == "dice":
        return dice_loss
    elif loss_name.lower() == "bce":
        return tf.keras.losses.BinaryCrossentropy()
    else:
        raise ValueError(f"Unsupported loss name: {loss_name}. Choose from 'bce_dice', 'dice', 'bce'.")

