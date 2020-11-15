"""
Dataset loading for Depth Completion KITTI Dataset
:ref:`http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion`
"""

from typing import Callable, Dict, Optional, Tuple, Union

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
        super().__init__(
            kitti_raw_root,
            depth_completion_root,
            subset,
            select_cams,
            select_calibs,
            imu_data,
            lidar_raw_data,
            transform,
            download,
        )

        for path in self._paths:
            del path["lidar"]
