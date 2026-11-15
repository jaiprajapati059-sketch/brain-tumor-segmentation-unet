"""
Unit Tests for Dataset Processing & Patient-Level Splitting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np
import pandas as pd

from src.dataset import extract_patient_id, split_dataset_by_patient


class TestDatasetProcessing(unittest.TestCase):
    """Test suite for patient extraction and group splitting guarantees."""

    def test_extract_patient_id_standard_tcga(self):
        """Verify extraction from standard TCGA filename."""
        path1 = "kaggle_3m/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_13.tif"
        path2 = "kaggle_3m/TCGA_DU_6404_19991206/TCGA_DU_6404_19991206_22_mask.tif"
        path3 = "data/TCGA_FG_5962_20010410_5.tif"

        self.assertEqual(extract_patient_id(path1), "TCGA_CS_4941")
        self.assertEqual(extract_patient_id(path2), "TCGA_DU_6404")
        self.assertEqual(extract_patient_id(path3), "TCGA_FG_5962")

    def test_extract_patient_id_generic_slice(self):
        """Verify extraction from generic patient identifier."""
        path = "data/PATIENT_0042_slice_15_mask.tif"
        self.assertEqual(extract_patient_id(path), "PATIENT_0042_slice")

    def test_strict_patient_level_splitting_zero_leakage(self):
        """CRITICAL: Ensure that train, val, and test partitions have 0% patient overlap."""
        # Generate mock cohort of 20 patients with multiple slices each
        records = []
        for p_idx in range(1, 21):
            p_id = f"TCGA_TEST_{p_idx:04d}"
            num_slices = np.random.randint(5, 15)
            for s_idx in range(1, num_slices + 1):
                records.append({
                    "image_path": f"data/{p_id}/{p_id}_{s_idx}.tif",
                    "mask_path": f"data/{p_id}/{p_id}_{s_idx}_mask.tif",
                    "patient_id": p_id,
                    "filename": f"{p_id}_{s_idx}.tif"
                })

        df = pd.DataFrame(records)

        train_df, val_df, test_df = split_dataset_by_patient(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            verbose=False
        )

        train_pts = set(train_df["patient_id"].unique())
        val_pts = set(val_df["patient_id"].unique())
        test_pts = set(test_df["patient_id"].unique())

        # 1. Total slice count preservation
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(df))

        # 2. Strict pairwise disjointness (ZERO PATIENT OVERLAP)
        self.assertTrue(train_pts.isdisjoint(val_pts), "Patient overlap detected between Train and Val!")
        self.assertTrue(train_pts.isdisjoint(test_pts), "Patient overlap detected between Train and Test!")
        self.assertTrue(val_pts.isdisjoint(test_pts), "Patient overlap detected between Val and Test!")

        # 3. Ratio fidelity
        total_pts = len(df["patient_id"].unique())
        self.assertAlmostEqual(len(train_pts) / total_pts, 0.70, delta=0.05)


if __name__ == "__main__":
    unittest.main()
