"""
Tests for test_lidar_point_cloud module
"""

import os

import numpy as np

from torch_kitti.raw import lidar_point_cloud as lpc


def test_load_lidar_point_cloud():
    path = os.path.join(os.path.dirname(__file__), "test_data", "lidar_data.bin")

    # homologous
    points = lpc.load_lidar_point_cloud(path, projective=True)
    assert np.all(points[:, -1] == 1)
