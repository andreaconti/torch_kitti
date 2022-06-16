import pytest
import torch
import torchvision.transforms.functional as F

from torch_kitti.depth_prediction import KittiDepthPredictionDataset


@pytest.mark.parametrize("load_stereo", [False, True])
def test_transform(raw_sync_rect_path, depth_completion_path, load_stereo):
    """
    Simple test using a custom transform
    """

    def to_tensor_crop(ex):
        # to tensor
        for label in ex.keys():
            ex[label] = F.to_tensor(ex[label])
        return ex

    ds = KittiDepthPredictionDataset(
        raw_sync_rect_path,
        depth_completion_path,
        subset="train",
        load_stereo=load_stereo,
        transform=to_tensor_crop,
    )

    ex = ds[0]
    for label in ex.keys():
        assert isinstance(ex[label], torch.Tensor)
