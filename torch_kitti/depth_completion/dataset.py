"""
Dataset loading for Depth Completion KITTI Dataset
:ref:`http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion`
"""

import errno
import os
import re
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing_extensions import Literal

from torch_kitti.depth_completion import download as kitti_depth_completion_download
from torch_kitti.depth_completion import (
    folders_check as kitti_depth_completion_folders_check,
)
from torch_kitti.raw.calibration import (
    CamCalib,
    load_imu_to_lidar,
    load_lidar_to_cam_00,
)
from torch_kitti.raw.inertial_measurement_unit import IMUData
from torch_kitti.raw.lidar_point_cloud import load_lidar_point_cloud
from torch_kitti.raw.synced_rectified import check_drives as kitti_raw_check_drives
from torch_kitti.raw.synced_rectified import download as kitti_raw_download

__all__ = ["KittiDepthCompletionDataset"]

_Cams = Union[
    Tuple[str], Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]
]

_Calibs = Union[
    Tuple[str], Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]
]


class KittiDepthCompletionDataset(Dataset):
    """
    2017 KITTI depth completion benchmarks
    dataset, consisting of 93k training and 1.5k test images.

    Ground truth has been acquired by accumulating 3D point clouds from a
    360 degree Velodyne HDL-64 Laserscanner and a consistency check using
    stereo camera pairs.

    To load this dataset are needed:

    KITTI Raw dataset
        can be found at :ref:`http://www.cvlibs.net/datasets/kitti/raw_data.php`
        and must be unzipped with the following folders structure:

    KITTI depth completion maps
        can be found at :ref:`http://www.cvlibs.net/datasets/kitti/eval_depth.php?\
        benchmark=depth_completion` and must be unzipped with the following structure
        (the same stated in the development kit)

    Parameters
    ----------
    kitti_raw_root: str
        path to the root of the KITTI raw data sync+rect folder.
    depth_completion_root: str
        path to the root of the depth completion path.
    subset: str, default train
        If 'train' creates the dataset from training set, if 'val' creates
        the dataset from validation set, if 'test' creates the
        dataset from test set.
    select_cams: Tuple[str], default ("cam_02",)
        images to be loaded among cam_00, cam_01, cam_02, cam_03.
    select_calibs: Tuple[str], default ("cam_00, "cam_02")
        calibration objects to be loaded among cam_00, cam_01, cam_02, cam_03, they
        are instances of :class:`torch_kitti.raw_calibration.CamCalib`.
    imu_data: bool, default False
        if load IMU Data, when true examples will contain "imu_data" and also
        "imu_to_lidar" fields.
    lidar_raw_data: str, optional
        if "projective" Lidar Data are loaded in the projective space
        (removing reflectance), if "reflectance" Lidar Data are loaded with
        the reflectance, if None Lidar Data are not loaded. When enabled examples
        will contain "lidar_data" and also "lidar_to_cam_00" fields.
    transform: Callable[[Dict], Dict], optional
        transformation applied to each output dictionary.
    download: bool, default False
        If true, downloads the dataset from the internet and puts
        it in root directories. If dataset is already downloaded,
        it is not downloaded again.

    .. note::
        On the test subset only the image, the groundtruth and the lidar image can be
        provided
    """

    def __init__(
        self,
        kitti_raw_root: str,
        depth_completion_root: str,
        subset: Literal["train", "val", "test"] = "train",
        select_cams: Optional[_Cams] = None,
        select_calibs: Optional[_Calibs] = None,
        imu_data: bool = False,
        lidar_raw_data: Literal[None, "projective", "reflectance"] = None,
        transform: Callable[[Dict], Dict] = None,
        download: bool = False,
    ):

        if download:
            # download depth completion dataset
            if not os.path.exists(depth_completion_root):
                kitti_depth_completion_download(depth_completion_root)
            elif os.path.isdir(depth_completion_root) and not os.listdir(
                depth_completion_root
            ):
                kitti_depth_completion_download(depth_completion_root)

            # download kitti raw rect+sync dataset
            if not os.path.exists(kitti_raw_root):
                kitti_raw_download(kitti_raw_root)
            elif os.path.isdir(kitti_raw_root) and not os.listdir(kitti_raw_root):
                kitti_raw_download(kitti_raw_root)

        # check folders
        if not kitti_depth_completion_folders_check(depth_completion_root):
            raise ValueError(f"path {depth_completion_root} contains wrong data")
        if not kitti_raw_check_drives(kitti_raw_root):
            raise ValueError(f"path {kitti_raw_root} contains wrong data")

        # params
        self._subset = subset
        self.transform = transform
        self._select_cams = list(select_cams) if select_cams is not None else []
        self._select_calibs = list(select_calibs) if select_calibs is not None else []
        self._imu_data = imu_data
        self._lidar_raw_data = lidar_raw_data

        # checks
        for cam in self._select_cams:
            if cam not in ["cam_00", "cam_01", "cam_02", "cam_03"]:
                raise ValueError("cam must be among cam_00, cam_01, cam_02, cam_03")
        for calib in self._select_calibs:
            if calib not in ["cam_00", "cam_01", "cam_02", "cam_03"]:
                raise ValueError("calib must be among cam_00, cam_01, cam_02, cam_03")
        if self._lidar_raw_data:
            if self._lidar_raw_data not in ["projective", "reflectance"]:
                raise ValueError("lidar_raw_data is projective or reflectance")

        # computing paths
        if subset in ["train", "val"]:

            depth_completion_path = os.path.join(depth_completion_root, subset)

            # find lidar and groundtruths
            all_files = []
            for dirpath, dirnames, filenames in os.walk(depth_completion_path):
                files = [os.path.join(dirpath, f) for f in filenames]
                all_files.extend(files)

            all_lidar_raws = [f for f in all_files if "velodyne_raw" in f]
            groundtruths = [f for f in all_files if "groundtruth" in f]

            # build matching image paths
            drive_regex = re.compile(r"[0-9]+_[0-9]+_[0-9]+_drive_[0-9]+_sync")
            date_regex = re.compile(r"[0-9]+_[0-9]+_[0-9]+")
            camera_regex = re.compile(r"image_[0-9]+")

            images = []
            for groundtruth_path in groundtruths:
                drive = drive_regex.findall(groundtruth_path)[0]
                date = date_regex.findall(groundtruth_path)[0]
                camera = camera_regex.findall(groundtruth_path)[0]
                image = groundtruth_path.split(os.path.sep)[-1]
                images.append(
                    os.path.join(kitti_raw_root, date, drive, camera, "data", image)
                )

            self._paths = [
                {"img": img, "gt": gt, "lidar": lidar}
                for img, gt, lidar in zip(images, groundtruths, all_lidar_raws)
            ]

            for path in self._paths:

                # append cams paths
                for cam in self._select_cams:
                    image = "image_{}".format(cam.split("_")[1].zfill(2))
                    path[cam] = re.sub(r"image_\d+", image, path["img"])

                # append selected calibs
                if self._select_calibs:
                    date = date_regex.findall(path["img"])[0]
                    path["cam2cam"] = os.path.join(
                        kitti_raw_root, date, "calib_cam_to_cam.txt"
                    )

                # append lidar data
                if self._lidar_raw_data is not None:
                    date = date_regex.findall(path["img"])[0]
                    path["lidar_raw_data"] = (
                        os.path.splitext(
                            re.sub(r"image_\d+", "velodyne_points", path["img"])
                        )[0]
                        + ".bin"
                    )
                    path["lidar_raw_to_cam_00"] = os.path.join(
                        kitti_raw_root, date, "calib_velo_to_cam.txt"
                    )

                # append imu data
                if self._imu_data:
                    date = date_regex.findall(path["img"])[0]
                    path["imu_data"] = (
                        os.path.splitext(re.sub(r"image_\d+", "oxts", path["img"]))[0]
                        + ".txt"
                    )
                    path["imu_to_lidar"] = os.path.join(
                        kitti_raw_root, date, "calib_imu_to_velo.txt"
                    )

        elif subset == "test":

            folders = os.path.join(
                depth_completion_root, "val_selection_cropped", "{0}"
            )
            files = [
                os.path.join(folders, f.replace("groundtruth_depth", "{0}"))
                for f in os.listdir(folders.format("groundtruth_depth"))
                if os.path.isfile(os.path.join(folders.format("groundtruth_depth"), f))
            ]

            self._paths = [
                {
                    "img": path.format("image"),
                    "gt": path.format("groundtruth_depth"),
                    "lidar": path.format("velodyne_raw"),
                }
                for path in files
            ]

        else:
            raise ValueError("subset must be train, val or test. Not {}".format(subset))

    def _open_img(self, path):
        extensions = [".png", ".jpg"]

        image = None
        for ext in extensions:
            try:
                image = Image.open(os.path.splitext(path)[0] + ext)
            except FileNotFoundError:
                pass

        if image is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

        return image

    def __getitem__(self, x):
        paths = self._paths[x]
        result = dict()

        # open image
        if "img" in paths:
            img = self._open_img(paths["img"])
            result["img"] = img

        # compute lidar points
        if "lidar" in paths:
            lidar = self._open_img(paths["lidar"])
            lidar = np.array(lidar).astype(np.float32) / 256.0
            lidar = np.expand_dims(lidar, axis=-1)
            result["lidar"] = lidar

        # compute groundtruth points
        if "gt" in paths:
            gt = self._open_img(paths["gt"])
            gt = np.array(gt).astype(np.float32) / 256.0
            gt = np.expand_dims(gt, axis=-1)
            result["gt"] = gt

        # load all the optional stuff :)
        if self._subset != "test":
            for cam in self._select_cams:
                result[cam] = self._open_img(paths[cam])
            for calib in self._select_calibs:
                idx = int(re.compile("[0-9]{2}").findall(calib)[0])
                calib_file = CamCalib.open(idx, paths["cam2cam"])
                result[f"cam_0{idx}_calib"] = calib_file
            if self._lidar_raw_data:
                result["lidar_raw_data"] = load_lidar_point_cloud(
                    paths["lidar_raw_data"], self._lidar_raw_data
                )
                result["lidar_raw_to_cam_00"] = load_lidar_to_cam_00(
                    paths["lidar_raw_to_cam_00"]
                )
            if self._imu_data:
                result["imu_data"] = IMUData.open(paths["imu_data"])
                result["imu_to_lidar"] = load_imu_to_lidar(paths["imu_to_lidar"])

        if self.transform is not None:
            result = self.transform(result)
        return result

    def __len__(self):
        return len(self._paths)
