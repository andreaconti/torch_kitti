"""
Tests over the KITTI Depth Completion Dataset
"""

import pytest

from torch_kitti.depth_completion import KittiDepthCompletionDataset

from ..dataset_utils import (
    get_depth_completion_path,
    get_sync_rect_path,
    on_depth_completion_dataset,
    on_sync_rect_dataset,
)


@on_sync_rect_dataset
@on_depth_completion_dataset
def test_dataset_train():

    # train subset
    ds = KittiDepthCompletionDataset(
        get_sync_rect_path(),
        get_depth_completion_path(),
        subset="train",
        load_stereo=True,
        load_previous=1,
    )

    assert len(ds) == 85898 // 2

    ex = ds[0]
    keys = [
        "img_left",
        "gt_left",
        "lidar_left",
        "intrinsics_left",
        "img_right",
        "gt_right",
        "lidar_right",
        "intrinsics_right",
        "img_previous_right",
        "gt_previous_right",
        "lidar_previous_right",
        "intrinsics_previous_right",
        "img_previous_left",
        "gt_previous_left",
        "lidar_previous_left",
        "intrinsics_previous_left",
    ]
    for key in keys:
        assert key in ex.keys()


def test_dataset_value_error():

    with pytest.raises(ValueError):
        KittiDepthCompletionDataset(
            "..",
            "..",
            subset="something_wrong",
        )

    with pytest.raises(ValueError):
        KittiDepthCompletionDataset("..", "..", load_previous=-23)
