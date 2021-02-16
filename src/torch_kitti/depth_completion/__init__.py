"""
Dataset loading for Depth Completion KITTI Dataset
:ref:`http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion`
"""

__all__ = ["KittiDepthCompletionDataset", "download", "folders_check"]

from ._download import download, folders_check
from .dataset import KittiDepthCompletionDataset
