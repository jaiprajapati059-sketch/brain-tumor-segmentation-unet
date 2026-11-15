"""
Project Configuration Module
Brain Tumor Semantic Segmentation - LGG MRI Dataset (Buda et al.)
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import os



@dataclass
class Config:
    """Centralized configuration for data loading, preprocessing, model architecture,
    training hyperparameters, and output paths.
    """

    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    # Raw dataset root: Supports standard Kaggle 'kaggle_3m' or subfolders
    data_dir: str = "data/kaggle_3m"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    best_model_name: str = "best_unet_lgg.keras"
    training_log_name: str = "training_history.csv"
    evaluation_results_name: str = "test_metrics.json"

    # --------------------------------------------------------------------------
    # Dataset & Preprocessing Hyperparameters
    # --------------------------------------------------------------------------
    image_height: int = 256
    image_width: int = 256
    num_channels: int = 3       # FLAIR/T1/T2 multi-channel TIFF or 3-channel RGB
    num_classes: int = 1        # Binary segmentation (Tumor vs Background)
    
    # --------------------------------------------------------------------------
    # Patient-Level Splitting (CRITICAL: Zero Patient Overlap)
    # --------------------------------------------------------------------------
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # --------------------------------------------------------------------------
    # Training Loop Hyperparameters
    # --------------------------------------------------------------------------
    batch_size: int = 16
    initial_learning_rate: float = 1e-4
    min_learning_rate: float = 1e-7
    epochs: int = 50
    patience_early_stopping: int = 12
    patience_reduce_lr: int = 5
    reduce_lr_factor: float = 0.5

    # --------------------------------------------------------------------------
    # Loss Formulation
    # --------------------------------------------------------------------------
    bce_weight: float = 1.0     # Alpha: Binary Cross-Entropy weight
    dice_weight: float = 1.0    # Beta: Dice Loss weight
    smooth_factor: float = 1e-6 # Numerical epsilon to prevent division by zero

    # --------------------------------------------------------------------------
    # Evaluation & Inference
    # --------------------------------------------------------------------------
    prediction_threshold: float = 0.5
    num_visual_samples: int = 5

    @property
    def model_save_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.best_model_name)

    @property
    def log_save_path(self) -> str:
        return os.path.join(self.output_dir, self.training_log_name)

    @property
    def eval_save_path(self) -> str:
        return os.path.join(self.output_dir, self.evaluation_results_name)

    def ensure_directories(self) -> None:
        """Create output and checkpoint directories if they do not exist."""
        for d in [self.checkpoint_dir, self.output_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration overrides from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to read YAML configs. Please run `pip install pyyaml`.")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def save_yaml(self, yaml_path: str) -> None:
        """Save current configuration to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save YAML configs. Please run `pip install pyyaml`.")
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

