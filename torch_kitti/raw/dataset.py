"""
PyTorch Dataset to load KITTI Raw Data
"""

import glob
import os
import re
from typing import Callable, Dict, Optional, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset

from .calibration import CamCalib, load_imu_to_lidar, load_lidar_to_cam_00
from .inertial_measurement_unit import IMUData
from .lidar_point_cloud import load_lidar_point_cloud

__all__ = ["KittiRawDataset"]

_Cams = Union[
    Tuple[str], Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]
]

_Calibs = Union[
    Tuple[str], Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]
]


class KittiRawDataset(Dataset):
    def __init__(
        self,
        root: str,
        select_cams: _Cams = ("cam_02",),
        imu_data: bool = False,
        lidar_data: Optional[str] = "projective",
        select_calibs: _Calibs = (
            "cam_00",
            "cam_02",
        ),
        transform: Optional[Callable[[Dict], Dict]] = None,
        download: bool = False,
    ):
        """
        Dataset loading KITTI Raw Data available, KITTI raw data are organized by date
        and drive and inside each drive are contained images taken by cameras,
        for example::

            .
            └── 2011_09_26
                ├── calib_cam_to_cam.txt
                ├── calib_imu_to_velo.txt
                ├── calib_velo_to_cam.txt
                └── 2011_09_26_drive_0002_sync
                    ├── image_00
                    │   └── data
                    │       └── ...
                    ├── image_01
                    │   └── data
                    │       └── ...
                    ├── image_02
                    │   └── data
                    │       └── ...
                    ├── image_03
                    │   └── data
                    │       └── ...
                    ├── oxts
                    │   └── data
                    │       └── ...
                    └── velodyne_points
                        └── data
                            └── ...

        Calibration data refers to the whole date.
        Each example is a dictionary composed by many entries, such entries
        can be selected at initialization time.

        #. cam_0X: PIL Image from camera X
        #. lidar_data: ndarray Nx4 containing lidar \
                       point cloud with respect to lidar coordinates
        #. lidar_to_cam_00: [R|T] matrix from lidar to cam_00 coordinates in \
                       projective space
        #. cam_0X_calib: :class:`torch_kitti.raw.calibration.CamCalib` object \
                       containing info about cam_0X calibration
        #. imu_data: :class:`torch_kitti.raw.inertial_measurement_unit.IMUData` \
                       object containing IMU data about the example
        #. imu_to_lidar: [R|T] matrix from imu to lidar coordinates in projective \
                       space

        Parameters
        ----------
        root: str
            path to the root of the dataset.
        select_cams: Tuple[str], default ("cam_02",)
            images to be loaded among cam_00, cam_01, cam_02, cam_03.
        imu_data: bool, default False
            if load IMU Data, when true examples will contain "imu_data" and also
            "imu_to_lidar" fields
        lidar_data: str, default "projective"
            if "projective" Lidar Data are loaded in the projective space
            (removing reflectance), if "reflectance" Lidar Data are loaded with
            the reflectance, if None Lidar Data are not loaded. When enabled examples
            will contain "lidar_data" and also "lidar_to_cam_00" fields
        select_calibs: Tuple[str], default ("cam_00, "cam_02")
            calibration objects to be loaded among cam_00, cam_01, cam_02, cam_03, they
            are instances of :class:`torch_kitti.raw_calibration.CamCalib`
        transform: Callable[[Dict], Dict], optional
            transformation applied to each output dictionary
        download: bool, default False
            if download the dataset in `root` path
        """

        # params
        self.select_cams = select_cams
        self.imu_data = imu_data
        self.lidar_data = lidar_data
        self.select_calibs = select_calibs

        if download is True:
            raise NotImplementedError()

        if lidar_data not in ["projective", "reflectance", None]:
            raise ValueError("lidar data must be 'projective', 'reflectance' or None")

        # load cam images
        cam_00 = list(
            filter(
                lambda f: "image_00" in f,
                glob.iglob(os.path.join(root, "**", "*.png"), recursive=True),
            )
        )
        cam_01 = [f.replace("image_00", "image_01", 1) for f in cam_00]
        cam_02 = [f.replace("image_00", "image_02", 1) for f in cam_00]
        cam_03 = [f.replace("image_00", "image_02", 1) for f in cam_00]

        # load lidar points
        lidar = [
            os.path.splitext(f.replace("image_00", "velodyne_points"))[0] + ".bin"
            for f in cam_00
        ]

        # load oxts
        oxts = [
            os.path.splitext(f.replace("image_00", "oxts"))[0] + ".txt" for f in cam_00
        ]

        # load calib files
        date_match = re.compile("[0-9]+_[0-9]+_[0-9]+")

        def extract_path(f, fname):
            date = date_match.findall(f)[0]
            return os.path.join(root, date, fname)

        cam2cam = [extract_path(f, "calib_cam_to_cam.txt") for f in cam_00]
        velo2cam = [extract_path(f, "calib_velo_to_cam.txt") for f in cam_00]
        imu2velo = [extract_path(f, "calib_imu_to_velo.txt") for f in cam_00]

        self._paths = [
            {
                "cam_00": c00,
                "cam_01": c01,
                "cam_02": c02,
                "cam_03": c03,
                "lidar_point_cloud": l_pc,
                "oxts": o,
                "cam2cam": c2c,
                "velo2cam": v2c,
                "imu2velo": i2v,
            }
            for c00, c01, c02, c03, l_pc, o, c2c, v2c, i2v in zip(
                cam_00, cam_01, cam_02, cam_03, lidar, oxts, cam2cam, velo2cam, imu2velo
            )
        ]

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, x):
        paths = self._paths[x]
        output = {}

        # load cams
        for cam in self.select_cams:
            output[cam] = Image.open(paths[cam])

        # load config files
        for calib in self.select_calibs:
            idx = int(re.compile("[0-9]{2}").findall(calib)[0])
            calib_file = CamCalib.open(idx, paths["cam2cam"])
            output[f"cam_0{idx}_calib"] = calib_file

        # load lidar points
        if self.lidar_data is not None:
            output["lidar_data"] = load_lidar_point_cloud(
                paths["lidar_point_cloud"], self.lidar_data == "projective"
            )
            output["lidar_to_cam_00"] = load_lidar_to_cam_00(paths["velo2cam"])

        # IMU Data
        if self.imu_data:
            output["imu_data"] = IMUData.open(paths["oxts"])
            output["imu_to_lidar"] = load_imu_to_lidar(paths["imu2velo"])

        return output
