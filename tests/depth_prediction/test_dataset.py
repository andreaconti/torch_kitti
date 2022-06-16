"""
Tests over the KITTI Depth Completion Dataset
"""

import pytest

from torch_kitti.depth_prediction import KittiDepthPredictionDataset
from torch_kitti.raw.calibration import CamCalib
import numpy as np


@pytest.mark.parametrize("subset", ["train", "val", "test", "all"])
def test_dataset_train(raw_sync_rect_path, depth_completion_path, subset):

    # train, val subset
    ds = KittiDepthPredictionDataset(
        raw_sync_rect_path,
        depth_completion_path,
        subset=subset,
        load_stereo=True,
    )

    assert len(ds) > 0

    ex = ds[0]
    keys = ["image", "gt", "intrinsics"]
    keys = [x + "_left" for x in keys] + [x + "_right" for x in keys]
    for key in keys:
        assert key in ex.keys()
        assert isinstance(ex[key], np.ndarray)


@pytest.mark.parametrize("subset", ["train", "val", "test"])
@pytest.mark.parametrize(
    "seq",
    [
        {"load_previous": 1},
        {"load_previous": (1, 4)},
        {"load_sequence": 3},
    ],
)
def test_dataset_sequences(raw_sync_rect_path, depth_completion_path, subset, seq):

    # train, val subset
    ds = KittiDepthPredictionDataset(
        raw_sync_rect_path,
        depth_completion_path,
        subset=subset,
        **seq,
    )

    assert len(ds) > 0

    ex = ds[0]
    keys = ["image", "gt", "intrinsics"]
    for key in keys:
        assert key in ex.keys()
        assert isinstance(ex[key], np.ndarray)
        if key in ["image", "gt"]:
            assert ex[key].ndim == 4
        if key == "intrinsics":
            assert ex[key].ndim == 3


def test_dataset_value_error(raw_sync_rect_path, depth_completion_path):

    with pytest.raises(ValueError):
        KittiDepthPredictionDataset(
            raw_sync_rect_path,
            depth_completion_path,
            subset="something_wrong",
        )

    with pytest.raises(ValueError):
        KittiDepthPredictionDataset(
            raw_sync_rect_path, depth_completion_path, load_previous=-23
        )
