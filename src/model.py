"""
U-Net Model Architecture for Medical Image Segmentation
Brain Tumor Semantic Segmentation - LGG MRI Dataset (Buda et al.)

References:
- Ronneberger, Fischer, & Brox (2015). "U-Net: Convolutional Networks for
  Biomedical Image Segmentation". MICCAI.
- Buda, Saha, & Mazurowski (2019). "Association of genomic subtypes of lower-grade
  gliomas with shape features on MRI". Radiology: AI.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(
    input_tensor: tf.Tensor,
    num_filters: int,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    name_prefix: str = "conv"
) -> tf.Tensor:
    """Standard U-Net Double Convolutional Block.
    
    Consists of:
      [Conv2D -> BatchNormalization -> ReLU] x 2
      + Optional Dropout for regularization.

    He normal initialization is used as it aligns with ReLU activations,
    preventing vanishing/exploding gradients during medical image training.
    """
    # First Convolutional sub-layer
    x = layers.Conv2D(
        filters=num_filters,
        kernel_size=(kernel_size, kernel_size),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv1"
    )(input_tensor)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)

    # Second Convolutional sub-layer
    x = layers.Conv2D(
        filters=num_filters,
        kernel_size=(kernel_size, kernel_size),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)

    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 1,
    base_filters: int = 32,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """Builds a classical U-Net architecture from scratch with explicitly linked
    skip connections.

    Architectural Overview:
      1. Contracting Path (Encoder):
         Progressively extracts deep semantic features by repeated convolution
         and downsampling (MaxPooling2D). Halves spatial dimensions at each level
         while doubling channel depth.
      2. Bottleneck:
         Captures the highest-level contextual representations at the lowest
         spatial resolution (e.g. 16x16). Includes dropout to curb co-adaptation.
      3. Expansive Path (Decoder):
         Progressively reconstructs spatial resolution via transposed convolutions
         (UpSampling / Conv2DTranspose).
      4. Skip Connections (CRITICAL FOR MEDICAL SEGMENTATION):
         Downsampling discards fine spatial coordinates. Skip connections concatenate
         high-resolution feature maps from encoder stages directly to corresponding
         decoder stages. This allows the decoder to recover delicate tumor boundaries,
         infiltrative margins, and sub-centimeter glioma lesions.
      5. Output Head:
         A 1x1 convolution with Sigmoid activation produces a per-pixel probability map
         ranging from 0.0 (healthy brain parenchyma) to 1.0 (hyperintense FLAIR tumor).

    Args:
        input_shape: Dimensions of input MRI slice (H, W, C), typically (256, 256, 3).
        num_classes: Number of segmentation target classes (1 for binary tumor mask).
        base_filters: Base number of filters at the first encoder stage (default: 32).
        dropout_rate: Dropout fraction applied in the bottleneck and deep encoder blocks.

    Returns:
        tf.keras.Model: Compiled or uncompiled Keras Model instance.
    """
    inputs = layers.Input(shape=input_shape, name="mri_input")

    # =========================================================================
    # 1. ENCODER (Contracting Path)
    # =========================================================================
    # Level 1: 256x256 -> Feature Map 1 (saved for Skip Connection 1)
    c1 = conv_block(inputs, num_filters=base_filters, name_prefix="enc_lvl1")
    p1 = layers.MaxPooling2D(pool_size=(2, 2), name="pool_lvl1")(c1)  # Output: 128x128

    # Level 2: 128x128 -> Feature Map 2 (saved for Skip Connection 2)
    c2 = conv_block(p1, num_filters=base_filters * 2, name_prefix="enc_lvl2")
    p2 = layers.MaxPooling2D(pool_size=(2, 2), name="pool_lvl2")(c2)  # Output: 64x64

    # Level 3: 64x64 -> Feature Map 3 (saved for Skip Connection 3)
    c3 = conv_block(p2, num_filters=base_filters * 4, name_prefix="enc_lvl3")
    p3 = layers.MaxPooling2D(pool_size=(2, 2), name="pool_lvl3")(c3)  # Output: 32x32

    # Level 4: 32x32 -> Feature Map 4 (saved for Skip Connection 4)
    c4 = conv_block(p3, num_filters=base_filters * 8, dropout_rate=dropout_rate, name_prefix="enc_lvl4")
    p4 = layers.MaxPooling2D(pool_size=(2, 2), name="pool_lvl4")(c4)  # Output: 16x16

    # =========================================================================
    # 2. BOTTLENECK (Bridge)
    # =========================================================================
    # Lowest spatial resolution (16x16) with richest contextual representation
    bottleneck = conv_block(
        p4,
        num_filters=base_filters * 16,
        dropout_rate=dropout_rate * 1.5,
        name_prefix="bottleneck"
    )

    # =========================================================================
    # 3. DECODER (Expansive Path with Skip Connections)
    # =========================================================================
    # Level 4 Decoder: Upsample 16x16 -> 32x32
    u4 = layers.Conv2DTranspose(
        filters=base_filters * 8,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="same",
        name="up_lvl4"
    )(bottleneck)
    # --- SKIP CONNECTION 4 ---
    # Concatenate upsampled representation with encoder feature map c4
    merge4 = layers.Concatenate(axis=-1, name="skip_conn_4")([u4, c4])
    c5 = conv_block(merge4, num_filters=base_filters * 8, name_prefix="dec_lvl4")

    # Level 3 Decoder: Upsample 32x32 -> 64x64
    u3 = layers.Conv2DTranspose(
        filters=base_filters * 4,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="same",
        name="up_lvl3"
    )(c5)
    # --- SKIP CONNECTION 3 ---
    # Concatenate upsampled representation with encoder feature map c3
    merge3 = layers.Concatenate(axis=-1, name="skip_conn_3")([u3, c3])
    c6 = conv_block(merge3, num_filters=base_filters * 4, name_prefix="dec_lvl3")

    # Level 2 Decoder: Upsample 64x64 -> 128x128
    u2 = layers.Conv2DTranspose(
        filters=base_filters * 2,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="same",
        name="up_lvl2"
    )(c6)
    # --- SKIP CONNECTION 2 ---
    # Concatenate upsampled representation with encoder feature map c2
    merge2 = layers.Concatenate(axis=-1, name="skip_conn_2")([u2, c2])
    c7 = conv_block(merge2, num_filters=base_filters * 2, name_prefix="dec_lvl2")

    # Level 1 Decoder: Upsample 128x128 -> 256x256
    u1 = layers.Conv2DTranspose(
        filters=base_filters,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="same",
        name="up_lvl1"
    )(c7)
    # --- SKIP CONNECTION 1 ---
    # Concatenate upsampled representation with shallow encoder feature map c1
    merge1 = layers.Concatenate(axis=-1, name="skip_conn_1")([u1, c1])
    c8 = conv_block(merge1, num_filters=base_filters, name_prefix="dec_lvl1")

    # =========================================================================
    # 4. FINAL OUTPUT HEAD
    # =========================================================================
    # 1x1 Convolution mapping the feature space to binary mask probability
    # Sigmoid activation guarantees output is bounded strictly in [0.0, 1.0]
    outputs = layers.Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        activation="sigmoid",
        dtype="float32",
        name="tumor_mask_output"
    )(c8)

    model = models.Model(inputs=inputs, outputs=outputs, name="UNet_LGG_BrainTumor")
    return model


if __name__ == "__main__":
    test_model = build_unet(input_shape=(256, 256, 3), num_classes=1, base_filters=32)
    test_model.summary()

