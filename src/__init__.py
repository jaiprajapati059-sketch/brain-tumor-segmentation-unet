"""
Brain Tumor Semantic Segmentation Package
LGG MRI Dataset (Buda et al.)
"""

from src.config import Config
from src.dataset import extract_patient_id, parse_lgg_dataset, split_dataset_by_patient

# Conditional deep learning imports (requires tensorflow)
try:
    from src.model import build_unet
    from src.losses import CombinedBCEDiceLoss, dice_loss, bce_dice_loss
    from src.metrics import dice_coef, iou_metric, precision_metric, recall_metric
    from src.dataset import load_and_preprocess_sample, LGGSequence
except ImportError:
    # TensorFlow not yet installed; allow utility and data parsing usage
    pass


