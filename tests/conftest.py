"""
Provides fixtures generating on the fly fake data
"""

import itertools
import os
import pathlib
import pickle
import shutil
import tempfile

import pytest

__all__ = ["raw_sync_rect_path", "depth_completion_path"]

test_data = os.path.join(os.path.dirname(__file__), "test_data")

# raw sync + rect


@pytest.fixture(scope="session")
def raw_sync_rect_path():

    # needed resources
    with open(os.path.join(test_data, "raw_date_drives.pkl"), "rb") as f:
        raw_date_drives = pickle.load(f)

    raw_files = range(10)

    cam2cam = os.path.join(test_data, "raw", "calib_cam_to_cam.txt")
    imu2velo = os.path.join(test_data, "raw", "calib_imu_to_velo.txt")
    velo2cam = os.path.join(test_data, "raw", "calib_velo_to_cam.txt")
    lidar_data = os.path.join(test_data, "raw", "lidar_data.bin")
    imu_data = os.path.join(test_data, "raw", "imu_data.txt")

    sync_rect_cams = [
        os.path.join(test_data, "raw", "sync_rect", f"cam_0{i}.png") for i in range(4)
    ]

    # dataset root
    root = os.path.join(tempfile.gettempdir(), "kitti_raw_sync_rect_root")

    # create images, oxts and lidar
    for date in raw_date_drives.keys():
        for drive in raw_date_drives[date]:
            for file_idx in raw_files:

                common_path = os.path.join(
                    root,
                    date,
                    date + "_drive_" + str(drive).zfill(4) + "_sync",
                    "{0}",
                    "data",
                    str(file_idx).zfill(10) + ".{1}",
                )

                # copy lidar
                lidar_path = common_path.format("velodyne_points", "bin")
                pathlib.Path(os.path.dirname(lidar_path)).mkdir(
                    parents=True, exist_ok=True
                )
                os.symlink(lidar_data, lidar_path)

                # copy oxts
                imu_path = common_path.format("oxts", "txt")
                pathlib.Path(os.path.dirname(imu_path)).mkdir(
                    parents=True, exist_ok=True
                )
                os.symlink(imu_data, imu_path)

                # copy cams
                for i in range(4):
                    img_path = common_path.format(f"image_0{i}", "png")
                    pathlib.Path(os.path.dirname(img_path)).mkdir(
                        parents=True, exist_ok=True
                    )
                    os.symlink(sync_rect_cams[i], img_path)

    # create calib
    for date in raw_date_drives.keys():
        common_path = os.path.join(root, date, "{0}")
        os.symlink(cam2cam, common_path.format("calib_cam_to_cam.txt"))
        os.symlink(velo2cam, common_path.format("calib_velo_to_cam.txt"))
        os.symlink(imu2velo, common_path.format("calib_imu_to_velo.txt"))

    # return dataset root path
    yield root

    # at the end delete
    shutil.rmtree(root)


# depth completion


@pytest.fixture(scope="session")
def depth_completion_path():

    # load dates and drives
    with open(
        os.path.join(test_data, "depth_completion_train_date_drives.pkl"), "rb"
    ) as f:
        train_date_drives = pickle.load(f)

    with open(
        os.path.join(test_data, "depth_completion_val_date_drives.pkl"), "rb"
    ) as f:
        val_date_drives = pickle.load(f)

    # dataset path
    root = os.path.join(tempfile.gettempdir(), "kitti_depth_completion_root")

    # create anonymous
    os.makedirs(os.path.join(root, "test_depth_completion_anonymous"))
    os.makedirs(os.path.join(root, "test_depth_prediction_anonymous"))

    # create train
    for date in train_date_drives.keys():
        for drive in train_date_drives[date]:
            for ftype, cam, file_idx in itertools.product(
                ["groundtruth", "velodyne_raw"], ["image_02", "image_03"], range(10)
            ):

                img_path = os.path.join(
                    root,
                    "train",
                    date + "_drive_" + str(drive).zfill(4) + "_sync",
                    "proj_depth",
                    ftype,
                    cam,
                    str(file_idx).zfill(10) + ".png",
                )

                pathlib.Path(os.path.dirname(img_path)).mkdir(
                    parents=True, exist_ok=True
                )
                os.symlink(
                    os.path.join(test_data, "depth_completion", f"{ftype}_{cam}.png"),
                    img_path,
                )

    # create val
    for date in val_date_drives.keys():
        for drive in val_date_drives[date]:
            for ftype, cam, file_idx in itertools.product(
                ["groundtruth", "velodyne_raw"], ["image_02", "image_03"], range(10)
            ):

                img_path = os.path.join(
                    root,
                    "val",
                    date + "_drive_" + str(drive).zfill(4) + "_sync",
                    "proj_depth",
                    ftype,
                    cam,
                    str(file_idx).zfill(10) + ".png",
                )

                pathlib.Path(os.path.dirname(img_path)).mkdir(
                    parents=True, exist_ok=True
                )
                os.symlink(
                    os.path.join(test_data, "depth_completion", f"{ftype}_{cam}.png"),
                    img_path,
                )

    # create test
    for date in val_date_drives.keys():
        for drive in val_date_drives[date]:
            for cam, file_idx in itertools.product(["image_02", "image_03"], range(10)):

                common_path = os.path.join(
                    root,
                    "val_selection_cropped",
                    "{0}",
                    date
                    + "_drive_"
                    + str(drive).zfill(4)
                    + "_sync_image_"
                    + str(file_idx).zfill(10)
                    + "_"
                    + cam
                    + ".{1}",
                )

                # save image
                img_path = common_path.format("image", "png")
                pathlib.Path(os.path.dirname(img_path)).mkdir(
                    parents=True, exist_ok=True
                )
                os.symlink(
                    os.path.join(test_data, "depth_completion", "test_image.png"),
                    img_path,
                )

                # save groundtruth
                gt_path = common_path.format("groundtruth_depth", "png")
                pathlib.Path(os.path.dirname(gt_path)).mkdir(
                    parents=True, exist_ok=True
                )
                os.symlink(
                    os.path.join(
                        test_data, "depth_completion", "groundtruth_image_02.png"
                    ),
                    gt_path,
                )

                # save velodyne_raw
                vl_path = common_path.format("velodyne_raw", "png")
                pathlib.Path(os.path.dirname(vl_path)).mkdir(
                    parents=True, exist_ok=True
                )
                os.symlink(
                    os.path.join(
                        test_data, "depth_completion", "velodyne_raw_image_02.png"
                    ),
                    vl_path,
                )

                # save intrinsics
                i_path = common_path.format("intrinsics", "txt")
                pathlib.Path(os.path.dirname(i_path)).mkdir(parents=True, exist_ok=True)
                os.symlink(
                    os.path.join(test_data, "depth_completion", "intrinsics.txt"),
                    i_path,
                )

    # return dataset root path
    yield root

    # at the end delete
    shutil.rmtree(root)
