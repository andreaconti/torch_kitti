"""
Tests for test_lidar_point_cloud module
"""

import glob
import os

import numpy as np

from torch_kitti.raw import lidar_point_cloud as lpc


def test_load_lidar_point_cloud(raw_sync_rect_path):

    path = glob.glob(os.path.join(raw_sync_rect_path, "**/*.bin"), recursive=True)[0]

    # homologous
    points = lpc.load_lidar_point_cloud(path, projective=True)
    assert np.all(points[:, -1] == 1)
