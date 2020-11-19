"""
Tests about the dataset loader
"""

import pytest

from torch_kitti.raw.dataset import KittiRawDataset
from torch_kitti.raw.synced_rectified import check_drives

# test dataset

keys_no_previous = {
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
    "lidar_data",
    "lidar_to_cam_00",
}

keys_previous = {
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
    "lidar_data",
    "lidar_to_cam_00",
    "cam_00_previous",
    "cam_01_previous",
    "cam_02_previous",
    "cam_03_previous",
    "cam_00_calib_previous",
    "cam_01_calib_previous",
    "cam_02_calib_previous",
    "cam_03_calib_previous",
    "imu_data_previous",
    "imu_to_lidar_previous",
    "lidar_data_previous",
    "lidar_to_cam_00_previous",
}


@pytest.mark.parametrize("transform", [None, lambda x: x])
@pytest.mark.parametrize(
    "load_previous, keys_expected",
    [
        (0, keys_no_previous),
        (1, keys_previous),
        ((1, 5), keys_previous),
    ],
)
def test_raw_dataset_loading(
    raw_sync_rect_path, load_previous, transform, keys_expected
):
    ds = KittiRawDataset(
        raw_sync_rect_path,
        select_cams=("cam_00", "cam_01", "cam_02", "cam_03"),
        imu_data=True,
        lidar_data="projective",
        select_calibs=("cam_00", "cam_01", "cam_02", "cam_03"),
        load_previous=load_previous,
        transform=transform,
    )

    assert len(ds) > 0
    ex = ds[0]
    ex_keys = set(ex.keys())
    assert ex_keys == keys_expected


# test folders


def test_check_drives(raw_sync_rect_path):
    assert check_drives(raw_sync_rect_path) is True
