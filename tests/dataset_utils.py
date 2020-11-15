"""
Provided utilities to handle datasets testing
"""

import functools
import os

import pytest

__all__ = [
    "on_sync_rect_dataset",
    "get_sync_rect_path",
    "on_depth_completion_dataset",
    "get_depth_completion_path",
]

# SYNC+RECT RAW DATASET


def on_sync_rect_dataset(f=None):

    # wrapper
    if f is not None:

        @pytest.mark.sync_rect_dataset
        def _wrapper(*args, **kwargs):
            if "KITTI_SYNC_RECT_ROOT" not in os.environ:
                pytest.skip(
                    "To execute tests on Kitti sync rect \
                    dataset export KITTI_SYNC_RECT_ROOT"
                )
            else:
                return f(*args, **kwargs)

        return functools.update_wrapper(_wrapper, f)

    # to skip a module
    to_skip = "KITTI_SYNC_RECT_ROOT" not in os.environ
    return pytest.mark.skipif(
        to_skip,
        reason="To execute tests on Kitti sync rect \
            dataset export KITTI_SYNC_RECT_ROOT",
    )


def get_sync_rect_path():
    return os.environ["KITTI_SYNC_RECT_ROOT"]


# DEPTH COMPLETION DATASET
def on_depth_completion_dataset(f=None):

    # wrapper
    if f is not None:

        @pytest.mark.depth_completion_dataset
        def _wrapper(*args, **kwargs):
            if "KITTI_DEPTH_COMPLETION_ROOT" not in os.environ:
                pytest.skip(
                    "To execute tests on Kitti sync rect \
                    dataset export KITTI_DEPTH_COMPLETION_ROOT"
                )
            else:
                return f(*args, **kwargs)

        return functools.update_wrapper(_wrapper, f)

    # to skip a module
    to_skip = "KITTI_DEPTH_COMPLETION_ROOT" not in os.environ
    return pytest.mark.skipif(
        to_skip,
        reason="To execute tests on Kitti sync rect \
            dataset export KITTI_DEPTH_COMPLETION_ROOT",
    )


def get_depth_completion_path():
    return os.environ["KITTI_DEPTH_COMPLETION_ROOT"]
