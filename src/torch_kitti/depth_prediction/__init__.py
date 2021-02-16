"""
Dataset loading for Depth Completion KITTI Dataset
:ref:`http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion`
"""

__all__ = ["KittiDepthPredictionDataset", "download", "folders_check"]

from ..depth_completion import download, folders_check
from .dataset import KittiDepthPredictionDataset
