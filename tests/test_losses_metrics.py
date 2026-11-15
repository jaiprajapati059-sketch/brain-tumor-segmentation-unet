"""
Unit Tests for Custom Segmentation Metrics & Loss Functions
"""

import unittest
import numpy as np

from src.metrics import numpy_dice_coefficient, numpy_iou


class TestMetricsAndLosses(unittest.TestCase):
    """Test mathematical correctness of overlap metrics and boundary conditions."""

    def test_dice_coefficient_identical_masks(self):
        """Identical foreground masks must yield a Dice score of 1.0."""
        mask_true = np.zeros((256, 256), dtype=np.float32)
        mask_true[50:100, 50:100] = 1.0

        mask_pred = mask_true.copy()
        score = numpy_dice_coefficient(mask_true, mask_pred)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_dice_coefficient_disjoint_masks(self):
        """Completely disjoint masks must yield approximately 0.0."""
        mask_true = np.zeros((256, 256), dtype=np.float32)
        mask_true[10:50, 10:50] = 1.0

        mask_pred = np.zeros((256, 256), dtype=np.float32)
        mask_pred[100:150, 100:150] = 1.0

        score = numpy_dice_coefficient(mask_true, mask_pred)
        self.assertLess(score, 0.001)

    def test_dice_coefficient_both_empty(self):
        """Both masks completely empty (healthy negative slice) should return 1.0."""
        mask_true = np.zeros((256, 256), dtype=np.float32)
        mask_pred = np.zeros((256, 256), dtype=np.float32)

        score = numpy_dice_coefficient(mask_true, mask_pred)
        self.assertEqual(score, 1.0)

    def test_iou_identical_masks(self):
        """Identical masks must yield an IoU of 1.0."""
        mask_true = np.zeros((256, 256), dtype=np.float32)
        mask_true[20:60, 20:60] = 1.0

        mask_pred = mask_true.copy()
        score = numpy_iou(mask_true, mask_pred)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_iou_partial_overlap(self):
        """Half overlap test: |A ∩ B| = 50, |A| = 100, |B| = 100 -> IoU = 50 / 150 = 0.333."""
        mask_true = np.zeros((10, 10), dtype=np.float32)
        mask_pred = np.zeros((10, 10), dtype=np.float32)

        mask_true[0:5, 0:5] = 1.0  # 25 pixels
        mask_pred[0:5, 0:5] = 1.0  # 25 pixels

        score = numpy_iou(mask_true, mask_pred)
        self.assertAlmostEqual(score, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
