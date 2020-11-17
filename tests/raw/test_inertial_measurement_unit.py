"""
IMU Data testing
"""

import glob
import os

from torch_kitti.raw.inertial_measurement_unit import IMUData


def test_imu_data(raw_sync_rect_path):

    path = glob.glob(
        os.path.join(raw_sync_rect_path, "**/0000000000.txt"), recursive=True
    )[0]

    imu_data = IMUData.open(path)

    for field in [
        "lat",
        "lon",
        "alt",
        "roll",
        "pitch",
        "yaw",
        "vn",
        "ve",
        "vf",
        "vl",
        "vu",
        "ax",
        "ay",
        "ay",
        "af",
        "al",
        "au",
        "wx",
        "wy",
        "wz",
        "wf",
        "wl",
        "wu",
        "pos_accuracy",
        "vel_accuracy",
        "navstat",
        "numsats",
        "posmode",
        "velmode",
        "orimode",
    ]:
        assert hasattr(imu_data, field)
