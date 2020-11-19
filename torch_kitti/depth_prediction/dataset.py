"""
Dataset loading for Depth Completion KITTI Dataset
:ref:`http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction`
"""

from typing import Callable, Dict, Tuple, Union

from typing_extensions import Literal

from ..depth_completion.dataset import KittiDepthCompletionDataset

_Cams = Union[
    Tuple[str], Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]
]

_Calibs = Union[
    Tuple[str], Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]
]


class KittiDepthPredictionDataset(KittiDepthCompletionDataset):
    """
    2017 KITTI depth prediction benchmarks
    dataset, consisting of 93k training and 1.5k test images.

    Ground truth has been acquired by accumulating 3D point clouds from a
    360 degree Velodyne HDL-64 Laserscanner and a consistency check using
    stereo camera pairs.

    To load this dataset are needed:

    KITTI Raw dataset (sync+rect)
        can be found at :ref:`http://www.cvlibs.net/datasets/kitti/raw_data.php`,
        automatically downloaded if required.

    KITTI depth prediction maps
        can be found at :ref:`http://www.cvlibs.net/datasets/kitti/eval_depth.php?\
        benchmark=depth_prediction`, automatically downloaded if required

    Parameters
    ----------
    kitti_raw_root: str
        path to the root of the KITTI raw data sync+rect folder.
    depth_prediction_root: str
        path to the root of the depth prediction path.
    subset: str, default train
        If 'train' creates the dataset from training set, if 'val' creates
        the dataset from validation set, if 'test' creates the
        dataset from test set.
    load_stereo: bool, default False
        if True each batch provides left and right synchronized and rectified
        cameras (the dataset size is halved). Not available on testing.
    load_previous: Union[int, Tuple[int, int]], optional
        if used a previous nth frame from the same sequence is provided, a random
        previous frame in the range (n, m) is choosen if provided a tuple. If that
        frame is not available is provided the same frame twice.
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
        depth_prediction_root: str,
        subset: Literal["train", "val", "test"] = "train",
        load_stereo: bool = False,
        load_previous: Union[Tuple[int, int], int] = 0,
        transform: Callable[[Dict], Dict] = lambda x: x,
        download: bool = False,
    ):
        super().__init__(
            kitti_raw_root,
            depth_prediction_root,
            subset,
            load_stereo,
            load_previous,
            transform,
            download,
        )

        for path in self._paths:
            del path["lidar"]
