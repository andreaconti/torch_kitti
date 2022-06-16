"""
Tests over the KITTI Depth Completion Dataset
"""

import pickle

import numpy as np
from torch_kitti.raw.calibration import CamCalib
import pytest

from torch_kitti.depth_completion import KittiDepthCompletionDataset, folders_check

# tests on the dataset


@pytest.mark.parametrize("subset", ["train", "val", "test", "all"])
def test_dataset(raw_sync_rect_path, depth_completion_path, subset):

    # train, val subset
    ds = KittiDepthCompletionDataset(
        raw_sync_rect_path,
        depth_completion_path,
        subset=subset,
        load_stereo=True,
    )

    assert len(ds) > 0

    ex = ds[0]
    keys = ["image", "lidar", "gt", "intrinsics"]
    keys = [x + "_left" for x in keys] + [x + "_right" for x in keys]
    for key in keys:
        assert key in ex.keys()
        assert isinstance(ex[key], np.ndarray)


def test_dataset_value_error(raw_sync_rect_path, depth_completion_path):

    with pytest.raises(ValueError):
        KittiDepthCompletionDataset(
            raw_sync_rect_path,
            depth_completion_path,
            subset="something_wrong",
        )


def test_pickable(raw_sync_rect_path, depth_completion_path):
    pickle.dumps(KittiDepthCompletionDataset(raw_sync_rect_path, depth_completion_path))


# test folder check


def test_folders_check(depth_completion_path):
    assert folders_check(depth_completion_path) is True
