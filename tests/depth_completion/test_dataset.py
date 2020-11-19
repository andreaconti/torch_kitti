"""
Tests over the KITTI Depth Completion Dataset
"""

import pickle

import pytest

from torch_kitti.depth_completion import KittiDepthCompletionDataset, folders_check

# tests on the dataset


@pytest.mark.parametrize("subset", ["train", "val"])
def test_dataset(raw_sync_rect_path, depth_completion_path, subset):

    ds = KittiDepthCompletionDataset(
        raw_sync_rect_path,
        depth_completion_path,
        subset=subset,
        load_stereo=True,
        load_previous=1,
    )

    assert len(ds) > 0

    ex = ds[0]
    keys = [
        "img_left",
        "gt_left",
        "lidar_left",
        "intrinsics_left",
        "img_right",
        "gt_right",
        "lidar_right",
        "intrinsics_right",
        "img_previous_right",
        "gt_previous_right",
        "lidar_previous_right",
        "intrinsics_previous_right",
        "img_previous_left",
        "gt_previous_left",
        "lidar_previous_left",
        "intrinsics_previous_left",
    ]
    for key in keys:
        assert key in ex.keys()


def test_dataset_test(raw_sync_rect_path, depth_completion_path):

    # test subset
    with pytest.raises(ValueError):
        ds = KittiDepthCompletionDataset(
            raw_sync_rect_path,
            depth_completion_path,
            subset="test",
            load_stereo=True,
        )

    with pytest.raises(ValueError):
        ds = KittiDepthCompletionDataset(
            raw_sync_rect_path,
            depth_completion_path,
            subset="test",
            load_previous=1,
        )

    ds = KittiDepthCompletionDataset(
        raw_sync_rect_path,
        depth_completion_path,
        subset="test",
    )

    assert len(ds) > 0

    ex = ds[0]
    keys = ["img", "gt", "intrinsics"]
    for key in keys:
        assert key in ex.keys()


def test_dataset_value_error(raw_sync_rect_path, depth_completion_path):

    with pytest.raises(ValueError):
        KittiDepthCompletionDataset(
            raw_sync_rect_path,
            depth_completion_path,
            subset="something_wrong",
        )


def test_pickable(raw_sync_rect_path, depth_completion_path):
    pickle.dumps(KittiDepthCompletionDataset(raw_sync_rect_path, depth_completion_path))


# test folder check


def test_folders_check(depth_completion_path):
    assert folders_check(depth_completion_path) is True
