"""
IMU Data testing
"""

import os

from torch_kitti.raw.inertial_measurement_unit import IMUData


def test_imu_data():
    path = os.path.join(os.path.dirname(__file__), "test_data", "imu_data.txt")
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
