"""
Utilities to handle calibration files
"""

from typing import Dict, Tuple

import numpy as np

__all__ = ["CamCalib", "load_imu_to_lidar", "load_lidar_to_cam_00"]

# LOADING CALIBRATION FILES


# cam_to_cam.txt


class CamCalib:
    def __init__(
        self,
        cam: int,
        image_size: Tuple[int, int],
        intrinsics: np.ndarray,
        distortion: np.ndarray,
        extrinsics: np.ndarray,
        rect_image_size: Tuple[int, int],
        rect_rotation: np.ndarray,
        projection_matrix: np.ndarray,
    ):
        """
        Calibration metrics for a single camera.

        Parameters
        ----------
        cam: int
            the choosen camera among 0, 1, 2, 3
        image_size: (int, int)
            size of the image before rectification
        intrinsics: ndarray
            3x3 array containing the intrinsics parameters
            of the camera
        distortion: ndarray
            k1, k2, p1, p2, k3 distortion coefficients where k1, k2 and k3 are
            the radial coeeficients. p1 and p2 are the tangential distortion
            coefficients
        extrinsics: ndarray
            4x4 array containing rototraslation matrix in the projective space,
            They seem to be a transformation from a common worl coordinate system
            into the camera's coordinate system
        rect_image_size: (int, int)
            size of the image after rectification
        rect_rotation: ndarray
            3x3 rotation matrix performing rectigying rotation for reference coordinate
            to make images of multiple cameras lie on the same plan
        projection_matrix: ndarray
            the projection matrix from 3D to 2D after rectification.
        """
        self.cam = cam
        self.image_size = image_size
        self.intrinsics = intrinsics
        self.distortion = distortion
        self.extrinsics = extrinsics
        self.rect_image_size = rect_image_size
        self.rect_rotation = rect_rotation
        self.projection_matrix = projection_matrix

    @staticmethod
    def _to_homologous_coord(rot_matrix):
        R = np.eye(4)
        R[:3, :3] = rot_matrix
        return R

    @staticmethod
    def open(cam: int, path: str) -> "CamCalib":

        # load data
        dict_values: Dict[str, np.ndarray] = dict()
        with open(path, "rt") as f:
            for line in f.readlines():
                name, vector = line.split(":", maxsplit=1)
                if name == "calib_time":
                    continue
                vector = np.array(list(map(float, vector.strip().split(" "))))
                dict_values[name] = vector

        # convert
        cams = "0" + str(cam)

        image_size = dict_values[f"S_{cams}"].astype(int)
        image_size = (int(image_size[0]), int(image_size[1]))

        rect_image_size = dict_values[f"S_rect_{cams}"].astype(int)
        rect_image_size = (int(rect_image_size[0]), int(rect_image_size[1]))

        R = dict_values[f"R_{cams}"].reshape(3, 3)
        T = dict_values[f"T_{cams}"]
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = T

        return CamCalib(
            cam=cam,
            image_size=image_size,
            intrinsics=dict_values[f"K_{cams}"].reshape(3, 3),
            distortion=dict_values[f"D_{cams}"],
            extrinsics=extrinsics,
            rect_image_size=rect_image_size,
            rect_rotation=CamCalib._to_homologous_coord(
                dict_values[f"R_rect_{cams}"].reshape(3, 3)
            ),
            projection_matrix=dict_values[f"P_rect_{cams}"].reshape(3, 4),
        )


# imu_to_velo.txt & calib_velo_to_cam.txt


def _load_rototraslation(path: str) -> np.ndarray:
    dict_values: Dict[str, np.ndarray] = dict()
    with open(path, "rt") as f:
        for line in f.readlines():
            name, vector = line.split(":", maxsplit=1)
            if name == "calib_time":
                continue
            vector = np.array(list(map(float, vector.strip().split(" "))))
            dict_values[name] = vector

    R = dict_values["R"].reshape(3, 3)
    T = dict_values["T"]
    rt = np.eye(4)
    rt[:3, :3] = R
    rt[:3, 3] = T

    return rt


def load_imu_to_lidar(path: str) -> np.ndarray:
    """
    returns the rototraslation matrix in projective coordinates from the
    inertial measurement unit to the lidar position. Done to read
    imu_to_velo.txt files.
    """
    return _load_rototraslation(path)


def load_lidar_to_cam_00(path: str) -> np.ndarray:
    """
    returns the rototraslation matrix in projective coordinates from the
    lidar to camera 00. Done to read velo_to_cam.txt files.
    """
    return _load_rototraslation(path)
