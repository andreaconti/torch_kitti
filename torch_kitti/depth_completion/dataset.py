"""
Dataset loading for Depth Completion KITTI Dataset
:ref:`http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion`
"""

import errno
import os
import random
import re
from typing import Callable, Dict, Tuple, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing_extensions import Literal

from torch_kitti.depth_completion import download as kitti_depth_completion_download
from torch_kitti.depth_completion import (
    folders_check as kitti_depth_completion_folders_check,
)
from torch_kitti.raw.calibration import CamCalib
from torch_kitti.raw.synced_rectified import check_drives as kitti_raw_check_drives
from torch_kitti.raw.synced_rectified import download as kitti_raw_download

__all__ = ["KittiDepthCompletionDataset"]

_Cams = Union[
    Tuple[str], Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]
]

_Calibs = Union[
    Tuple[str], Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]
]


def _identity(x: Dict) -> Dict:
    return x


class KittiDepthCompletionDataset(Dataset):
    """
    2017 KITTI depth completion benchmarks
    dataset, consisting of 93k training and 1.5k test images.

    Ground truth has been acquired by accumulating 3D point clouds from a
    360 degree Velodyne HDL-64 Laserscanner and a consistency check using
    stereo camera pairs.

    To load this dataset are needed:

    KITTI Raw dataset (sync+rect)
        can be found at :ref:`http://www.cvlibs.net/datasets/kitti/raw_data.php`,
        automatically downloaded if required.

    KITTI depth completion maps
        can be found at :ref:`http://www.cvlibs.net/datasets/kitti/eval_depth.php?\
        benchmark=depth_completion`, automatically downloaded if required

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
    load_stereo: bool, default False
        if True each batch provides left and right synchronized and rectified
        cameras (the dataset size is halved). Not available on testing.
    load_previous: Union[int, Tuple[int, int]], optional
        if used a previous nth frame from the same sequence is provided, a random
        previous frame in the range (n, m) is choosen if provided a tuple. if such
        frame cannot be found the same frame is returned as previous
    transform: Callable[[Dict], Dict], optional
        transformation applied to each output dictionary.
    download: bool, default False
        If true, downloads the dataset from the internet and puts
        it in root directories. If dataset is already downloaded,
        it is not downloaded again.

    .. note::
        On the test subset `load_stereo` and `load_previous` can not be used
    """

    def __init__(
        self,
        kitti_raw_root: str,
        depth_completion_root: str,
        subset: Literal["train", "val", "test"] = "train",
        load_stereo: bool = False,
        load_previous: Union[Tuple[int, int], int] = 0,
        transform: Callable[[Dict], Dict] = _identity,
        download: bool = False,
    ):

        # PARAMS

        self._subset = subset
        self._load_stereo = load_stereo
        self._load_previous = load_previous
        self.transform = transform

        if subset not in ["train", "val", "test"]:
            raise ValueError("subset must be in train, val, test")
        if not isinstance(load_previous, int) and not isinstance(load_previous, tuple):
            raise ValueError("load_previous int or tuple")
        if isinstance(load_previous, int) and load_previous < 0:
            raise ValueError("load_previous >= 0")

        # DOWNLOAD

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
                try:
                    drive = drive_regex.findall(groundtruth_path)[0]
                    date = date_regex.findall(groundtruth_path)[0]
                    camera = camera_regex.findall(groundtruth_path)[0]
                    image = groundtruth_path.split(os.path.sep)[-1]
                    images.append(
                        os.path.join(kitti_raw_root, date, drive, camera, "data", image)
                    )
                except IndexError:
                    raise RuntimeError("wrong path {}".format(groundtruth_path))

            # compute intrinsics
            cam2cam = []
            for img in images:
                date = date_regex.findall(img)[0]
                cam2cam.append(
                    os.path.join(kitti_raw_root, date, "calib_cam_to_cam.txt")
                )

            self._paths = [
                {"img": img, "gt": gt, "lidar": lidar, "cam2cam": c2c}
                for img, gt, lidar, c2c in zip(
                    images, groundtruths, all_lidar_raws, cam2cam
                )
            ]

        elif subset == "test":

            folders = os.path.join(
                depth_completion_root, "val_selection_cropped", "{0}"
            )
            files = [
                os.path.join(folders, f.replace("groundtruth_depth", "{1}"))
                for f in os.listdir(folders.format("groundtruth_depth"))
                if os.path.isfile(os.path.join(folders.format("groundtruth_depth"), f))
            ]

            self._paths = [
                {
                    "img": path.format("image", "image"),
                    "gt": path.format("groundtruth_depth", "groundtruth_depth"),
                    "lidar": path.format("velodyne_raw", "velodyne_raw"),
                    "intrinsics": os.path.splitext(path.format("intrinsics", "image"))[
                        0
                    ]
                    + ".txt",
                }
                for path in files
            ]

        else:
            raise ValueError("subset must be train, val or test. Not {}".format(subset))

        if load_stereo and subset == "test":
            raise ValueError("on test data stereo is not available")
        elif load_stereo:
            for path in self._paths:
                for key in path:
                    path[key] = re.sub("image_0[23]", "{0}", path[key])
            self._paths = list({v["img"]: v for v in self._paths}.values())

        if (
            isinstance(load_previous, (tuple))
            or isinstance(load_previous, int)
            and load_previous > 0
        ) and subset == "test":
            raise ValueError("on test data previous frame is not available")

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

    def _getitem(self, paths, _can_recur=True):
        result = dict()

        cam_used = int(re.match(".*image_([0-9]+).*", paths["img"]).groups()[0])

        # open image
        img = self._open_img(paths["img"])
        result["img"] = img

        # compute lidar points
        if "lidar" in paths:
            lidar = self._open_img(paths["lidar"])
            lidar = np.array(lidar).astype(np.float32) / 256.0
            lidar = np.expand_dims(lidar, axis=-1)
            result["lidar"] = lidar

        # compute groundtruth points
        gt = self._open_img(paths["gt"])
        gt = np.array(gt).astype(np.float32) / 256.0
        gt = np.expand_dims(gt, axis=-1)
        result["gt"] = gt

        # compute intrinsics
        if "cam2cam" in paths:
            calib = CamCalib.open(cam_used, paths["cam2cam"])
            result["intrinsics"] = calib.projection_matrix[:, :-1]
        elif "intrinsics" in paths:
            with open(paths["intrinsics"], "rt") as f:
                result["intrinsics"] = np.array(
                    list(map(float, f.readline().strip().split(" ")))
                ).reshape(3, 3)

        # compute previous
        if isinstance(self._load_previous, int) and self._load_previous > 0:
            curr_img_idx = int(
                re.match(r".*([0-9]{10}).png$", paths["img"]).groups()[0]
            )
            previous = max(0, curr_img_idx - self._load_previous)
            if not os.path.exists(
                re.sub(r"\d{10}", str(previous).zfill(10), paths["gt"])
            ):
                previous = curr_img_idx

        elif isinstance(self._load_previous, tuple):
            curr_img_idx = int(
                re.match(r".*([0-9]{10}).png$", paths["img"]).groups()[0]
            )
            previous = max(0, curr_img_idx - random.randint(*self._load_previous))
            if not os.path.exists(
                re.sub(r"\d{10}", str(previous).zfill(10), paths["gt"])
            ):
                previous = curr_img_idx

        else:
            previous = None

        if previous is not None and _can_recur:
            idx = str(previous).zfill(10)
            new_paths = dict()
            for k in paths:
                new_paths[k] = re.sub(r"[0-9]{10}", idx, paths[k])
            previous = self._getitem(new_paths, _can_recur=False)
            for k in previous:
                result[k + "_previous"] = previous[k]

        return result

    def __getitem__(self, x):

        if self._load_stereo:
            # load left
            left_paths = dict(**self._paths[x])
            for k in left_paths:
                left_paths[k] = left_paths[k].format("image_02")
            left_ex = self._getitem(left_paths)

            # load right
            right_paths = dict(**self._paths[x])
            for k in right_paths:
                right_paths[k] = right_paths[k].format("image_03")
            right_ex = self._getitem(right_paths)

            # rename and return
            left_ex_, right_ex_ = dict(), dict()
            for k_l, k_r in zip(left_ex.keys(), right_ex.keys()):
                left_ex_[k_l + "_left"] = left_ex[k_l]
                right_ex_[k_r + "_right"] = right_ex[k_r]

            return self.transform(dict(**left_ex_, **right_ex_))
        else:
            return self.transform(self._getitem(self._paths[x]))

    def __len__(self):
        return len(self._paths)
