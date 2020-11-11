"""
Tests about the dataset loader
"""

from torch_kitti.raw.dataset import KittiRawDataset

from ..dataset_utils import get_sync_rect_path, on_sync_rect_dataset


@on_sync_rect_dataset
def test_ds_loading():
    ds = KittiRawDataset(get_sync_rect_path())
    ds[0]
