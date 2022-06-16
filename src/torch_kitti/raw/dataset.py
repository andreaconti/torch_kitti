"""
PyTorch Dataset to load KITTI Raw Data
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch_kitti.common import DataElem, DataGroup, GenericDataset, _LoadPrev
from typing_extensions import Literal

from .synced_rectified import check_drives as sync_rect_check_drives
from .synced_rectified import download as sync_rect_download

__all__ = ["KittiRawDataset"]


def _identity(x: Dict) -> Dict:
    return x


def generate_examples(
    root: str,
    select_cams: List[int] = [2],
    imu_data: bool = False,
    lidar_data: Optional[str] = "projective",
    select_calibs: List[int] = [0, 2],
    load_previous: Union[Tuple[int, int], int] = 0,
    load_sequence: int = 1,
) -> List[DataGroup]:

    # find velodyne
    path = Path(root)
    all_velodyne = [
        DataElem("lidar_pcd", "pcd", path, cam=False, pcd_format=lidar_data)
        for path in path.glob("**/velodyne_points/data/*.bin")
    ]

    # add info required
    elems = []
    for vel in all_velodyne:
        ex_elems = [vel]

        ex_elems.append(
            DataElem(
                "lidar_to_cam_00",
                "rt",
                path / vel.date / "calib_velo_to_cam.txt",
                cam=False,
                drive=vel.drive,
                idx=vel.idx,
            )
        )

        ex_elems.append(
            DataElem(
                "imu_to_lidar",
                "rt",
                path / vel.date / "calib_imu_to_velo.txt",
                cam=False,
                drive=vel.drive,
                idx=vel.idx,
            )
        )

        for cam in select_cams:
            ex_elems.append(
                DataElem(
                    f"cam_{cam:0>2}",
                    "image",
                    str(vel.path.as_posix())
                    .replace("velodyne_points", f"image_{cam:0>2}")
                    .replace(".bin", ".png"),
                )
            )

        if imu_data:
            ex_elems.append(
                DataElem(
                    f"imu_data",
                    "imu",
                    str(vel.path.as_posix())
                    .replace("velodyne_points", "oxts")
                    .replace(".bin", ".txt"),
                    cam=False,
                )
            )

        for calib in select_calibs:
            ex_elems.append(
                DataElem(
                    f"cam_{calib:0>2}_calib",
                    "calib",
                    path / vel.date / "calib_cam_to_cam.txt",
                    cam=calib,
                    drive=vel.drive,
                    idx=vel.idx,
                )
            )

        _load_prev_ = _LoadPrev(vel)
        ex_elems = [
            _load_prev_(elem, load_previous, load_sequence) for elem in ex_elems
        ]
        elems.append(DataGroup(ex_elems))
    return elems


class KittiRawDataset(GenericDataset):
    def __init__(
        self,
        root: str,
        select_cams: List[int] = [2],
        imu_data: bool = False,
        lidar_data: Optional[str] = "projective",
        select_calibs: List[int] = [0, 2],
        load_previous: Union[Tuple[int, int], int] = 0,
        load_sequence: int = 1,
        transform: Callable[[Dict], Dict] = _identity,
        download: Union[bool, Literal["sync+rect"]] = False,
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
                    │   └── data
                    │       └── ...
                    ├── image_01
                    │   └── data
                    │       └── ...
                    ├── image_02
                    │   └── data
                    │       └── ...
                    ├── image_03
                    │   └── data
                    │       └── ...
                    ├── oxts
                    │   └── data
                    │       └── ...
                    └── velodyne_points
                        └── data
                            └── ...

        Calibration data refers to the whole date.
        Each example is a dictionary composed by many entries, such entries
        can be selected at initialization time.

        #. cam_0X: ndarray Image from camera X
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
        load_previous: Union[int, Tuple[int, int]], optional
            if used a previous nth frame from the same sequence is provided, a random
            previous frame in the range (n, m) is choosen if provided a tuple.
        load_sequence: int, optional
            It loads a sequence of frames, stacking them into a np.ndarray new dimension
            or in a list.
        transform: Callable[[Dict], Dict], optional
            transformation applied to each output dictionary
        download: bool or sync+rect, default False
            if True downloads the sync+rect version, otherwise the version of
            the dataset can be provided but by now only "sync+rect" is available
        """

        # PARAMS

        self.select_cams = set(select_cams)
        self.select_calibs = set(select_calibs)
        self.imu_data = imu_data
        self.transform = transform
        self._load_previous = load_previous
        self.lidar_data = lidar_data

        # CHECK params

        for cam in self.select_cams:
            if cam not in range(4):
                raise ValueError("each cam must be in range 0-3")

        for calib in self.select_calibs:
            if calib not in range(4):
                raise ValueError("each calib must be in range 0-3")

        if lidar_data not in ["projective", "reflectance", None]:
            raise ValueError("lidar data must be 'projective', 'reflectance' or None")

        if not isinstance(load_previous, (tuple, int)):
            raise ValueError(
                "load_previous must be and integer of a 2-tuple of integers"
            )

        if download not in [False, True, "sync+rect"]:
            raise ValueError("download in True, False or sync+rect")

        # DOWNLOAD

        if download is True or download == "sync+rect":

            # download sync+rect folders
            if not os.path.exists(root):
                sync_rect_download(root)
            elif os.path.isdir(root) and not os.listdir(root):
                sync_rect_download(root)

        # check folders
        if not sync_rect_check_drives(root):
            raise ValueError(f"path {root} contains wrong data")

        # PATHS

        elems = generate_examples(
            root,
            select_cams,
            imu_data,
            lidar_data,
            select_calibs,
            load_previous,
            load_sequence,
        )
        super().__init__(elems)
