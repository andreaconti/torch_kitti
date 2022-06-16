"""
Tests about the dataset loader
"""

import pickle

import pytest

from torch_kitti.raw.dataset import KittiRawDataset
from torch_kitti.raw.synced_rectified import check_drives

# test dataset

keys_expected = {
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
    "lidar_pcd",
    "lidar_to_cam_00",
}


@pytest.mark.parametrize(
    "opts",
    [
        {"load_previous": 0},
        {"load_previous": 1},
        {"load_previous": (1, 5)},
        {"load_sequence": 3},
    ],
)
def test_raw_dataset_loading(raw_sync_rect_path, opts):
    ds = KittiRawDataset(
        raw_sync_rect_path,
        select_cams=[0, 1, 2, 3],
        imu_data=True,
        lidar_data="projective",
        select_calibs=[0, 1, 2, 3],
        **opts,
    )

    assert len(ds) > 0
    ex = ds[0]
    ex_keys = set(ex.keys())
    assert ex_keys == keys_expected


def test_pickable(raw_sync_rect_path):

    ds = KittiRawDataset(
        raw_sync_rect_path,
        select_cams=[0, 1, 2, 3],
        imu_data=True,
        lidar_data="projective",
        select_calibs=[0, 1, 2, 3],
    )

    ds = pickle.dumps(ds)
    ds = pickle.loads(ds)


# test folders


def test_check_drives(raw_sync_rect_path):
    assert check_drives(raw_sync_rect_path) is True
