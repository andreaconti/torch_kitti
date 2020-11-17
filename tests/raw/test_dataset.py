"""
Tests about the dataset loader
"""

from torch_kitti.raw.dataset import KittiRawDataset


def test_raw_dataset_loading(raw_sync_rect_path):
    ds = KittiRawDataset(
        raw_sync_rect_path,
        select_cams=("cam_00", "cam_01", "cam_02", "cam_03"),
        imu_data=True,
        lidar_data="projective",
        select_calibs=("cam_00", "cam_01", "cam_02", "cam_03"),
    )

    assert len(ds) > 0
    ex = ds[0]
    ex_keys = set(ex.keys())
    keys = {
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
    assert ex_keys == keys
