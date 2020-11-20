import pytest
import torch
import torchvision.transforms.functional as F

from torch_kitti.depth_completion import KittiDepthCompletionDataset


@pytest.mark.parametrize(
    "load_previous, load_stereo, labels",
    [
        (0, False, ["img", "gt", "lidar"]),
        (
            1,
            False,
            ["img", "gt", "lidar", "img_previous", "gt_previous", "lidar_previous"],
        ),
        (
            1,
            True,
            [
                "img_left",
                "gt_left",
                "lidar_left",
                "img_right",
                "gt_right",
                "lidar_right",
                "img_previous_right",
                "gt_previous_right",
                "lidar_previous_right",
                "img_previous_left",
                "gt_previous_left",
                "lidar_previous_left",
            ],
        ),
    ],
)
def test_transform(
    raw_sync_rect_path, depth_completion_path, labels, load_previous, load_stereo
):
    """
    Simple test using a custom transform
    """

    def to_tensor_crop(ex):
        # to tensor
        for label in labels:
            ex[label] = F.to_tensor(ex[label])[:, :50, :50]
        return ex

    ds = KittiDepthCompletionDataset(
        raw_sync_rect_path,
        depth_completion_path,
        subset="train",
        load_previous=load_previous,
        load_stereo=load_stereo,
        transform=to_tensor_crop,
    )

    ex = ds[0]
    for label in labels:
        print(label)
        assert isinstance(ex[label], torch.Tensor)
        c, h, w = ex[label].shape
        assert c == 1 or c == 3
        assert h == 50
        assert w == 50
