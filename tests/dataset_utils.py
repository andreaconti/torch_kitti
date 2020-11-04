"""
Provided utilities to handle datasets testing
"""

import functools
import os

import pytest

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
