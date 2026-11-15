"""
Unit Tests for U-Net Model Architecture
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest

try:
    import tensorflow as tf
    from src.model import build_unet
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


class TestUNetArchitecture(unittest.TestCase):
    """Test suite for U-Net layer shapes and forward pass."""

    @unittest.skipUnless(_TF_AVAILABLE, "TensorFlow not installed in current test environment")
    def test_unet_output_shape(self):
        """Verify U-Net accepts (256, 256, 3) and produces (256, 256, 1) probability map."""
        model = build_unet(input_shape=(256, 256, 3), num_classes=1, base_filters=16)
        dummy_input = tf.zeros((1, 256, 256, 3), dtype=tf.float32)
        output = model(dummy_input)

        self.assertEqual(output.shape, (1, 256, 256, 1))

    @unittest.skipUnless(_TF_AVAILABLE, "TensorFlow not installed in current test environment")
    def test_sigmoid_activation_bounds(self):
        """Verify output values are bounded strictly within [0.0, 1.0]."""
        model = build_unet(input_shape=(256, 256, 3), num_classes=1, base_filters=16)
        dummy_input = tf.random.normal((1, 256, 256, 3))
        output = model(dummy_input)

        self.assertTrue(tf.reduce_all(output >= 0.0))
        self.assertTrue(tf.reduce_all(output <= 1.0))


if __name__ == "__main__":
    unittest.main()

