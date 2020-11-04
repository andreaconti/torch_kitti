"""
Test of calibration files loading
"""

import os

from torch_kitti.raw import calibration as calib


def test_open_cam_calib():
    path = os.path.join(os.path.dirname(__file__), "test_data", "calib_cam_to_cam.txt")

    for i in range(4):
        cam_i = calib.CamCalib.open(i, path)
        assert cam_i.cam == i


def test_load_imu_to_lidar():
    path = os.path.join(os.path.dirname(__file__), "test_data", "calib_imu_to_velo.txt")
    rt = calib.load_imu_to_lidar(path)
    assert rt.shape == (4, 4)


def test_load_lidar_to_cam_00():
    path = os.path.join(os.path.dirname(__file__), "test_data", "calib_velo_to_cam.txt")
    rt = calib.load_lidar_to_cam_00(path)
    assert rt.shape == (4, 4)
