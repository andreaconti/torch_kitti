"""
Utilities to load IMU Data
"""

import numpy as np

__all__ = ["IMUData"]


class IMUData:
    def __init__(self, **kwargs):
        for kw, v in kwargs.items():
            setattr(self, kw, v)

    @staticmethod
    def open(path: str) -> "IMUData":
        fields = [
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
        ]
        with open(path, "rt") as f:
            vector = f.readline()
            vector = np.array(list(map(float, vector.strip().split(" "))))

        return IMUData(**dict(zip(fields, vector)))
