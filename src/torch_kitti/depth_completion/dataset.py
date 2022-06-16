"""
Dataset loading for Depth Completion KITTI Dataset
:ref:`http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion`
"""

import os
from typing import Callable, Dict, List, Tuple, Union
import random

from typing_extensions import Literal

from torch_kitti.depth_completion import download as kitti_depth_completion_download
from torch_kitti.depth_completion import (
    folders_check as kitti_depth_completion_folders_check,
)
from torch_kitti.raw.synced_rectified import check_drives as kitti_raw_check_drives
from torch_kitti.raw.synced_rectified import download as kitti_raw_download
from torch_kitti.common import DataElem, DataGroup, GenericDataset, _LoadPrev

from pathlib import Path

__all__ = ["KittiDepthCompletionDataset"]


def _identity(x: Dict) -> Dict:
    return x


def _list_groundtruths(
    kitti_completion_root: str,
    subset: Literal["train", "val", "test", "all"],
    load_stereo: bool = False,
) -> List[DataElem]:

    path = Path(kitti_completion_root)
    if not load_stereo or subset == "test":
        gt_glob = "**/groundtruth/**/*.png"
    else:
        gt_glob = "**/groundtruth/**/image_02/**/*.png"

    gts = []
    if subset in ["train", "all", "test"]:
        used_path = path / "train"
        gts.extend(DataElem("gt", "depth", path) for path in used_path.glob(gt_glob))
    if subset in ["val", "all", "test"]:
        used_path = path / "val"
        gts.extend(DataElem("gt", "depth", path) for path in used_path.glob(gt_glob))

    if subset == "test":
        path = path / "val_selection_cropped/groundtruth_depth"
        test_gts = [DataElem("gt", "depth", path) for path in path.glob("*")]
        gts = [gt for gt in gts if gt in test_gts]

    return gts


def generate_examples(
    kitti_raw_root: str,
    kitti_completion_root: str,
    subset: Literal["train", "val", "test", "all"] = "train",
    load_stereo: bool = False,
    load_previous: Union[Tuple[int, int], int] = 0,
    load_sequence: int = 1,
) -> List[DataGroup]:

    if load_previous != 0 and load_sequence != 1:
        assert ValueError("can't use load_previous and load_sequence together")

    gts = _list_groundtruths(kitti_completion_root, subset, load_stereo)

    def _load_stereo(elem: DataElem):
        if load_stereo:
            path_left = str(elem.path.as_posix()).replace(
                f"image_{elem.cam:0>2}", f"image_02"
            )
            elem_left = DataElem(
                elem.name + "_left", elem.type, path_left, 2, elem.drive, elem.idx
            )
            path_right = str(elem.path.as_posix()).replace(
                f"image_{elem.cam:0>2}", f"image_03"
            )
            elem_right = DataElem(
                elem.name + "_right", elem.type, path_right, 3, elem.drive, elem.idx
            )
            return [elem_left, elem_right]
        return [elem]

    # load elems for each gt elem

    elems = []
    for gt in gts:
        ex_elems = []
        ex_elems.extend(_load_stereo(gt))

        data = "_".join(gt.drive.split("_")[:3])
        path = (
            Path(kitti_raw_root)
            / data
            / gt.drive
            / f"image_{gt.cam:0>2}/data/{gt.idx:0>10}.png"
        )
        img = DataElem("image", "image", path)
        ex_elems.extend(_load_stereo(img))

        path = Path(str(gt.path).replace("groundtruth", "velodyne_raw"))
        lidar = DataElem("lidar", "depth", path)
        ex_elems.extend(_load_stereo(lidar))

        path = Path(kitti_raw_root) / data / "calib_cam_to_cam.txt"
        calib = DataElem("intrinsics", "intrinsics", path, gt.cam, gt.drive, idx=gt.idx)
        ex_elems.extend(_load_stereo(calib))

        _load_prev_ = _LoadPrev(gt)
        ex_elems = [
            _load_prev_(elem, load_previous, load_sequence) for elem in ex_elems
        ]

        elems.append(DataGroup(ex_elems))
    return elems


class KittiDepthCompletionDataset(GenericDataset):
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
    load_sequence: int, optional
        if used loads a sequence of ``load_sequence`` frames, incompatible with
        ``load_previous``, if enough previous frames are not available the last
        one is used instead to fill the gap.
    load_sequence: int, optional
        It loads a sequence of frames, stacking them into a np.ndarray new dimension
        or in a list.
    transform: Callable[[Dict], Dict], optional
        transformation applied to each output dictionary.
    download: bool, default False
        If true, downloads the dataset from the internet and puts
        it in root directories. If dataset is already downloaded,
        it is not downloaded again.
    """

    def __init__(
        self,
        kitti_raw_root: str,
        kitti_completion_root: str,
        subset: Literal["train", "val", "test", "all"] = "train",
        load_stereo: bool = False,
        load_previous: Union[Tuple[int, int], int] = 0,
        load_sequence: int = 1,
        transform: Callable[[Dict], Dict] = _identity,
        download: bool = False,
    ):
        # checks
        if subset not in ["train", "val", "test", "all"]:
            raise ValueError(f"subset {subset} not in train, val, test, all")
        if isinstance(load_previous, int) and load_previous < 0:
            raise ValueError(f"load_previous negative not allowed")
        if isinstance(load_previous, tuple):
            if len(load_previous) != 2 or load_previous[0] < 0 or load_previous[1] < 0:
                raise ValueError(
                    f"load_previous must be a 2-tuple of positive ints or a single integer"
                )
        if load_sequence <= 0:
            raise ValueError(f"load_sequence <= 0 not allowed")

        # PARAMS
        self._subset = subset
        self._load_stereo = load_stereo
        self._load_previous = load_previous

        # DOWNLOAD
        if download:
            # download depth completion dataset
            if not os.path.exists(kitti_completion_root):
                kitti_depth_completion_download(kitti_completion_root)
            elif os.path.isdir(kitti_completion_root) and not os.listdir(
                kitti_completion_root
            ):
                kitti_depth_completion_download(kitti_completion_root)

            # download kitti raw rect+sync dataset
            if not os.path.exists(kitti_raw_root):
                kitti_raw_download(kitti_raw_root)
            elif os.path.isdir(kitti_raw_root) and not os.listdir(kitti_raw_root):
                kitti_raw_download(kitti_raw_root)

        # check folders
        if not kitti_depth_completion_folders_check(kitti_completion_root):
            raise ValueError(f"path {kitti_completion_root} contains wrong data")
        if not kitti_raw_check_drives(kitti_raw_root):
            raise ValueError(f"path {kitti_raw_root} contains wrong data")

        # SET PATHS
        elems = generate_examples(
            kitti_raw_root,
            kitti_completion_root,
            subset,
            load_stereo=load_stereo,
            load_previous=load_previous,
            load_sequence=load_sequence,
        )
        super().__init__(elems, transform)
