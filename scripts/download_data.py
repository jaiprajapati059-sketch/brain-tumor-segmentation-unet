"""
Dataset Download Helper Script
Brain Tumor Semantic Segmentation - LGG MRI Dataset (Buda et al.)
"""

import os
import sys
import subprocess
from pathlib import Path


def download_lgg_dataset(target_dir: str = "data"):
    """Downloads and unzips the LGG MRI segmentation dataset from Kaggle."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LGG MRI SEGMENTATION DATASET DOWNLOADER (Buda et al.)")
    print("=" * 70)
    print(f"Target directory: {target_path.resolve()}\n")

    # Check if kaggle CLI is installed
    try:
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Kaggle CLI not detected or kaggle.json not configured.")
        print("\nManual Download Instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation")
        print("2. Click 'Download' (archive is ~3 GB).")
        print(f"3. Extract contents into: {target_path.resolve()}/kaggle_3m/")
        print("=" * 70)
        return

    print("Kaggle CLI detected. Initiating download: mateuszbuda/lgg-mri-segmentation...")
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "mateuszbuda/lgg-mri-segmentation",
        "-p",
        str(target_path),
        "--unzip"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\nDownload and extraction completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Download failed with error: {e}")
        print("Please verify your ~/.kaggle/kaggle.json API token.")


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "data"
    download_lgg_dataset(dest)
