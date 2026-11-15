"""
Data Processing & Patient-Level Splitting Module
Brain Tumor Semantic Segmentation - LGG MRI Dataset (Buda et al.)

Crucial Medical CV Requirement:
Strict patient-level splitting to avoid spatial/anatomical data leakage between
training, validation, and test sets.
"""

import os
import re
import glob
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Generator

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split


def extract_patient_id(file_path: str) -> str:
    """Extracts a unique Patient Identifier from an LGG MRI file path or directory name.

    In the Buda et al. / TCGA-LGG dataset, files follow patterns such as:
      kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_13.tif
      TCGA_CS_4941_19960909_13_mask.tif

    To guarantee ZERO patient leakage, we extract the core patient/case identifier
    (e.g., 'TCGA_CS_4941' or 'TCGA_CS_4941_19960909'). Slices from the same patient
    share identical anatomical skull structures, ventricular morphology, and tissue
    density; grouping strictly by patient ensures generalizable evaluation.

    Args:
        file_path: Absolute or relative path to MRI slice or mask.

    Returns:
        patient_id: Standardized patient identifier string.
    """
    path = Path(file_path)
    stem = path.stem.replace("_mask", "")

    # 1. First attempt: match standard TCGA patient format (TCGA_XX_XXXX)
    tcga_match = re.match(r"(TCGA_[A-Z0-9]{2}_[A-Z0-9]{4})", stem)
    if tcga_match:
        return tcga_match.group(1)

    # 2. Second attempt: match directory name if formatted as patient folder
    parent_name = path.parent.name
    tcga_parent = re.match(r"(TCGA_[A-Z0-9]{2}_[A-Z0-9]{4})", parent_name)
    if tcga_parent:
        return tcga_parent.group(1)

    # 3. Third attempt: extract everything before the final slice index (e.g. 'PATIENT_01_slice')
    slice_idx_match = re.match(r"^(.*?)(?:_\d+)$", stem)
    if slice_idx_match:
        return slice_idx_match.group(1)

    # Fallback to parent directory name or stem
    return parent_name if parent_name not in ["", "kaggle_3m", "data", "raw"] else stem


def parse_lgg_dataset(dataset_dir: str) -> pd.DataFrame:
    """Recursively searches the dataset directory to locate all paired MRI image
    slices and corresponding ground truth segmentation masks.

    Args:
        dataset_dir: Root directory containing patient folders or image files.

    Returns:
        pd.DataFrame: Table with columns ['image_path', 'mask_path', 'patient_id', 'filename']

    Raises:
        FileNotFoundError: If no paired slices and masks are discovered.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{dataset_dir}' does not exist. "
            "Please ensure data is downloaded or run dataset.create_dummy_dataset()."
        )

    # Search for all mask files across supported formats (.tif, .tiff, .png, .jpg)
    mask_patterns = [
        str(dataset_path / "**" / "*_mask.tif"),
        str(dataset_path / "**" / "*_mask.tiff"),
        str(dataset_path / "**" / "*_mask.png"),
        str(dataset_path / "**" / "*_mask.jpg"),
    ]

    mask_files: List[str] = []
    for pattern in mask_patterns:
        mask_files.extend(glob.glob(pattern, recursive=True))

    records: List[Dict[str, str]] = []
    for mask_path in mask_files:
        # Determine expected raw image path by stripping '_mask'
        mask_p = Path(mask_path)
        img_filename = mask_p.name.replace("_mask", "")
        img_path = mask_p.parent / img_filename

        if img_path.exists():
            patient_id = extract_patient_id(str(img_path))
            records.append({
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_p.resolve()),
                "patient_id": patient_id,
                "filename": img_filename,
            })

    if not records:
        raise FileNotFoundError(
            f"No paired MRI slices found under '{dataset_dir}'. "
            "Expected files ending with '_mask.tif' and corresponding '.tif' images."
        )

    df = pd.DataFrame(records)
    return df


def split_dataset_by_patient(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Performs strict Patient-Level Splitting (Group Split) to guarantee no patient
    overlap between training, validation, and testing sets.

    Medical Computer Vision Justification:
        Random slice-level splitting introduces massive data leakage. Because consecutive
        axial slices from the same MRI volume share near-identical brain morphology, a
        slice-level split allows the network to 'cheat' by recognizing patient anatomy.
        Patient-level splitting guarantees that the model is evaluated on completely
        unseen human subjects, reflecting true clinical generalization.

    Args:
        df: DataFrame containing image metadata with 'patient_id' column.
        train_ratio: Fraction of patients allocated to training (default: 0.70).
        val_ratio: Fraction of patients allocated to validation (default: 0.15).
        test_ratio: Fraction of patients allocated to test (default: 0.15).
        random_seed: Random seed for deterministic reproducibility.
        verbose: Whether to print split summary statistics.

    Returns:
        (train_df, val_df, test_df): DataFrames for each partition.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    assert np.isclose(total_ratio, 1.0), f"Ratios must sum to 1.0 (got {total_ratio})"

    unique_patients = np.array(sorted(df["patient_id"].unique()))
    num_patients = len(unique_patients)

    if num_patients < 3:
        raise ValueError(
            f"At least 3 distinct patients are required for Train/Val/Test split. "
            f"Found only {num_patients} patients."
        )

    # First split: Separate Training set from (Validation + Test)
    val_test_ratio = val_ratio + test_ratio
    train_patients, val_test_patients = train_test_split(
        unique_patients,
        test_size=val_test_ratio,
        random_state=random_seed,
        shuffle=True
    )

    # Second split: Separate Validation from Test
    relative_test_ratio = test_ratio / val_test_ratio
    val_patients, test_patients = train_test_split(
        val_test_patients,
        test_size=relative_test_ratio,
        random_state=random_seed,
        shuffle=True
    )

    # Verify zero patient overlap (Disjoint sets)
    train_set, val_set, test_set = set(train_patients), set(val_patients), set(test_patients)
    assert train_set.isdisjoint(val_set), "CRITICAL: Patient leakage detected between Train and Val!"
    assert train_set.isdisjoint(test_set), "CRITICAL: Patient leakage detected between Train and Test!"
    assert val_set.isdisjoint(test_set), "CRITICAL: Patient leakage detected between Val and Test!"

    # Filter original slice dataframe
    train_df = df[df["patient_id"].isin(train_set)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_set)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_set)].reset_index(drop=True)

    if verbose:
        print("=" * 68)
        print("STRICT PATIENT-LEVEL SPLIT SUMMARY (Zero Data Leakage)")
        print("=" * 68)
        print(f"Total Unique Patients : {num_patients}")
        print(f"Total MRI Slices      : {len(df)}")
        print("-" * 68)
        print(f"Train Split : {len(train_patients):3d} patients ({len(train_df):5d} slices | {len(train_df)/len(df)*100:.1f}%)")
        print(f"Val Split   : {len(val_patients):3d} patients ({len(val_df):5d} slices | {len(val_df)/len(df)*100:.1f}%)")
        print(f"Test Split  : {len(test_patients):3d} patients ({len(test_df):5d} slices | {len(test_df)/len(df)*100:.1f}%)")
        print("=" * 68)

    return train_df, val_df, test_df


def load_and_preprocess_sample(
    image_path: str,
    mask_path: str,
    target_size: Tuple[int, int] = (256, 256),
    augment: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads an MRI image and corresponding mask, applies resizing, normalization,
    and optional geometric data augmentation.

    Args:
        image_path: Path to the raw MRI slice (.tif or other image format).
        mask_path: Path to the binary segmentation mask.
        target_size: Target (height, width) dimensions (default: 256x256).
        augment: If True, applies random horizontal/vertical flips and rotations.

    Returns:
        image: Preprocessed float32 image array of shape (H, W, 3), scaled to [0, 1].
        mask: Preprocessed float32 binary mask array of shape (H, W, 1), values {0.0, 1.0}.
    """
    # 1. Load MRI Image (3 channels)
    # OpenCV imread handles TIFF format. Convert BGR to RGB.
    raw_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if raw_img is None:
        raise ValueError(f"Failed to read MRI image at: {image_path}")
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # 2. Load Segmentation Mask (1 channel, Grayscale)
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        raise ValueError(f"Failed to read MRI mask at: {mask_path}")

    # 3. Resize
    # Bilinear interpolation for image (smooth intensity transitions)
    # Nearest-neighbor interpolation for mask (preserves strict binary 0/1 labels)
    resized_img = cv2.resize(raw_img, target_size, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(raw_mask, target_size, interpolation=cv2.INTER_NEAREST)

    # 4. Optional Data Augmentation (for Training Split)
    if augment:
        # Random horizontal flip
        if np.random.rand() > 0.5:
            resized_img = np.fliplr(resized_img)
            resized_mask = np.fliplr(resized_mask)

        # Random vertical flip
        if np.random.rand() > 0.5:
            resized_img = np.flipud(resized_img)
            resized_mask = np.flipud(resized_mask)

        # Random 90-degree rotation (orthogonal rotation preserves aspect ratio)
        k = np.random.choice([0, 1, 2, 3])
        if k > 0:
            resized_img = np.rot90(resized_img, k=k)
            resized_mask = np.rot90(resized_mask, k=k)

    # 5. Normalization & Type Conversion
    # Normalize pixel intensities to [0.0, 1.0]
    image = resized_img.astype(np.float32) / 255.0

    # Ensure mask is strictly binary {0.0, 1.0} and has shape (H, W, 1)
    mask = (resized_mask > 127).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return image, mask


class LGGSequence(tf.keras.utils.Sequence):
    """Memory-efficient, multithread-safe Keras Sequence for batch generation."""

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 16,
        target_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        shuffle: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_images = []
        batch_masks = []

        for idx in batch_indexes:
            row = self.df.iloc[idx]
            img, mask = load_and_preprocess_sample(
                image_path=row["image_path"],
                mask_path=row["mask_path"],
                target_size=self.target_size,
                augment=self.augment
            )
            batch_images.append(img)
            batch_masks.append(mask)

        return np.array(batch_images, dtype=np.float32), np.array(batch_masks, dtype=np.float32)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indexes)


def create_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = 16,
    target_size: Tuple[int, int] = (256, 256),
    augment: bool = False,
    shuffle: bool = True
) -> tf.data.Dataset:
    """Creates a high-performance tf.data.Dataset pipeline with prefetching."""

    image_paths = df["image_path"].values.tolist()
    mask_paths = df["mask_path"].values.tolist()

    def py_generator():
        indices = np.arange(len(image_paths))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            img, mask = load_and_preprocess_sample(
                image_paths[i],
                mask_paths[i],
                target_size=target_size,
                augment=augment
            )
            yield img, mask

    output_signature = (
        tf.TensorSpec(shape=(target_size[0], target_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(target_size[0], target_size[1], 1), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(py_generator, output_signature=output_signature)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def create_dummy_dataset(
    output_dir: str = "data/kaggle_3m",
    num_patients: int = 5,
    slices_per_patient: int = 12,
    image_size: Tuple[int, int] = (256, 256)
) -> pd.DataFrame:
    """Generates a synthetic LGG MRI dataset mimicking Buda et al.'s TCGA structure.

    Used for unit testing, CI pipelines, and immediate code verification before
    downloading the full ~3GB Kaggle dataset.

    Creates realistic synthetic axial brain slices with elliptical skull cavities,
    brain tissue textures, and simulated hyperintense FLAIR tumor regions.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    records = []

    print(f"Creating synthetic LGG dataset at: {output_dir}")
    print(f"Generating {num_patients} patients with {slices_per_patient} slices each...")

    for p_idx in range(1, num_patients + 1):
        # Format matching TCGA patient ID convention
        patient_id = f"TCGA_CS_{4940 + p_idx}_20000101"
        patient_folder = out_path / patient_id
        patient_folder.mkdir(parents=True, exist_ok=True)

        # Determine tumor presence across slices (tumors appear in central slices)
        center_slice = slices_per_patient // 2

        for s_idx in range(1, slices_per_patient + 1):
            img_filename = f"{patient_id}_{s_idx}.tif"
            mask_filename = f"{patient_id}_{s_idx}_mask.tif"

            img_file = patient_folder / img_filename
            mask_file = patient_folder / mask_filename

            # 1. Base synthetic MRI slice
            h, w = image_size
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            mask_canvas = np.zeros((h, w), dtype=np.uint8)

            # Skull boundary (outer ellipse)
            cv2.ellipse(canvas, (w // 2, h // 2), (w // 2 - 25, h // 2 - 15), 0, 0, 360, (60, 60, 60), -1)
            # Brain parenchyma (inner brain tissue)
            cv2.ellipse(canvas, (w // 2, h // 2), (w // 2 - 35, h // 2 - 25), 0, 0, 360, (140, 140, 140), -1)
            # Add anatomical noise/texture
            noise = np.random.normal(0, 10, (h, w, 3)).astype(np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # 2. Simulate hyperintense tumor region if near the center slice
            dist_from_center = abs(s_idx - center_slice)
            if dist_from_center <= 3:
                tumor_radius = max(8, 28 - dist_from_center * 6)
                tumor_center = (w // 2 + 35, h // 2 - 20)
                
                # Draw hyperintense tumor on MRI
                cv2.circle(canvas, tumor_center, tumor_radius, (245, 245, 245), -1)
                # Draw ground truth mask
                cv2.circle(mask_canvas, tumor_center, tumor_radius, 255, -1)

            # Save as TIFF
            cv2.imwrite(str(img_file), canvas)
            cv2.imwrite(str(mask_file), mask_canvas)

            records.append({
                "image_path": str(img_file.resolve()),
                "mask_path": str(mask_file.resolve()),
                "patient_id": patient_id,
                "filename": img_filename,
            })

    df = pd.DataFrame(records)
    print(f"Successfully generated {len(df)} synthetic MRI slices across {num_patients} patients.")
    return df

