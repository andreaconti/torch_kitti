"""
testing integration between torchvision and transformations
"""

import torch
import torchvision.transforms as V

import torch_kitti.transforms as K


def test_random_crop():

    # simulate input
    fake_img_1 = torch.randn(1, 600, 600)
    fake_img_2 = fake_img_1.clone()
    x = {"img_left": fake_img_1, "img_right": fake_img_2}

    output = K.functional.apply_to_features(V.RandomCrop([200, 200]), x)
    assert torch.all(output["img_left"] == output["img_right"])
