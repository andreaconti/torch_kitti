"""
Tests over the KITTI Depth Prediction Dataset
"""

import pytest

from torch_kitti.depth_prediction import KittiDepthPredictionDataset

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
    ds = KittiDepthPredictionDataset(
        get_sync_rect_path(),
        get_depth_completion_path(),
        subset="train",
        select_cams=("cam_00", "cam_01", "cam_02", "cam_03"),
        select_calibs=("cam_00", "cam_01", "cam_02", "cam_03"),
        imu_data=True,
        lidar_raw_data="projective",
    )

    assert len(ds) == 85898

    ex = ds[0]
    keys = [
        "img",
        "gt",
        "lidar_raw_data",
        "lidar_raw_to_cam_00",
        "cam_00",
        "cam_01",
        "cam_02",
        "cam_03",
        "cam_00_calib",
        "cam_01_calib",
        "cam_02_calib",
        "cam_03_calib",
        "imu_data",
        "imu_to_lidar",
    ]
    for key in keys:
        assert key in ex.keys()


def test_dataset_value_error():

    with pytest.raises(ValueError):
        KittiDepthPredictionDataset(
            "..",
            "..",
            select_cams=("image_01"),
        )

    with pytest.raises(ValueError):
        KittiDepthPredictionDataset(
            "..",
            "..",
            select_calibs=("image_01"),
        )

    with pytest.raises(ValueError):
        KittiDepthPredictionDataset(
            "..",
            "..",
            lidar_raw_data="something_wrong",
        )
