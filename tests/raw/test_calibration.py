"""
Test of calibration files loading
"""

import glob
import os

from torch_kitti.raw import calibration as calib


def test_open_cam_calib(raw_sync_rect_path):
    cam2cam = glob.glob(
        os.path.join(raw_sync_rect_path, "**/calib_cam_to_cam.txt"), recursive=True
    )[0]

    for i in range(4):
        cam_i = calib.CamCalib.open(i, cam2cam)
        assert cam_i.cam == i


def test_load_imu_to_lidar(raw_sync_rect_path):

    path = glob.glob(
        os.path.join(raw_sync_rect_path, "**/calib_imu_to_velo.txt"), recursive=True
    )[0]

    rt = calib.load_imu_to_lidar(path)
    assert rt.shape == (4, 4)


def test_load_lidar_to_cam_00(raw_sync_rect_path):

    path = glob.glob(
        os.path.join(raw_sync_rect_path, "**/calib_velo_to_cam.txt"), recursive=True
    )[0]

    rt = calib.load_lidar_to_cam_00(path)
    assert rt.shape == (4, 4)
