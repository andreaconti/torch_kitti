"""
Tests over the KITTI Depth Completion Dataset
"""

import pytest

from torch_kitti.depth_prediction import KittiDepthPredictionDataset


def test_dataset_train(raw_sync_rect_path, depth_completion_path):

    # train subset
    ds = KittiDepthPredictionDataset(
        raw_sync_rect_path,
        depth_completion_path,
        subset="train",
        load_stereo=True,
        load_previous=1,
    )

    assert len(ds) > 0

    ex = ds[0]
    keys = [
        "img_left",
        "gt_left",
        "intrinsics_left",
        "img_right",
        "gt_right",
        "intrinsics_right",
        "img_previous_right",
        "gt_previous_right",
        "intrinsics_previous_right",
        "img_previous_left",
        "gt_previous_left",
        "intrinsics_previous_left",
    ]
    for key in keys:
        assert key in ex.keys()


def test_dataset_value_error():

    with pytest.raises(ValueError):
        KittiDepthPredictionDataset(
            "..",
            "..",
            subset="something_wrong",
        )

    with pytest.raises(ValueError):
        KittiDepthPredictionDataset("..", "..", load_previous=-23)
