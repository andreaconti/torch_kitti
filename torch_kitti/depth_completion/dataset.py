"""
Dataset loading for Depth Completion KITTI Dataset
:ref:`http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion`
"""

import errno
import os
import re
from typing import Callable, Dict

import numpy as np
from PIL import Image

from torch_kitti.raw.dataset import KittiRawDataset

__all__ = ["KittiDepthCompletionDataset"]


class KittiDepthCompletionDataset(KittiRawDataset):
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
    download: bool, default False
        If true, downloads the dataset from the internet and puts
        it in root directories. If dataset is already downloaded,
        it is not downloaded again.
    """

    def __init__(
        self,
        kitti_raw_root: str,
        depth_completion_root: str,
        subset: str = "train",
        transform: Callable[[Dict], Dict] = None,
        download: bool = False,
    ):

        if download:
            raise NotImplementedError()

        # save transforms
        self.transform = transform

        if subset in ["train", "val"]:

            depth_completion_path = os.path.join(depth_completion_root, subset)

            # find lidar and groundtruths
            all_files = []
            for dirpath, dirnames, filenames in os.walk(depth_completion_path):
                files = [os.path.join(dirpath, f) for f in filenames]
                all_files.extend(files)

            all_lidar_raws = [f for f in all_files if "velodyne_raw" in f]
            groundtruths = [f for f in all_files if "groundtruth" in f]
            self.groundtruths = groundtruths

            # find matching image paths
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

        elif subset == "test":

            folders = os.path.join(
                depth_completion_path, "val_selection_cropped", "{0}"
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

        # open image
        img = self._open_img(paths["img"])

        # compute raw lidar points
        lidar = self._open_img(paths["lidar"])
        lidar = np.array(lidar).astype(np.float32) / 256.0
        lidar = np.expand_dims(lidar, axis=-1)

        # compute groundtruth points
        gt = self._open_img(paths["gt"])
        gt = np.array(gt).astype(np.float32) / 256.0
        gt = np.expand_dims(gt, axis=-1)

        result = {"img": img, "gt": gt, "lidar": lidar}
        if self.transform is not None:
            result = self.transform(result)
        return result

    def __len__(self):
        return len(self._paths)
