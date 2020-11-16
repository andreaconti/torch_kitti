import random

import numpy as np

import torch_kitti.transforms.functional as F
from torch_kitti.transforms import ApplyToFeatures


def test_apply_to_features():

    # test functional
    d = {"a": 1, "b": 1}

    d_ = F.apply_to_features(["a", "b"], lambda x: x + random.random(), d)
    assert d_["a"] == d_["b"]

    d_ = F.apply_to_features(
        ["a", "b"], lambda x: x + random.random(), d, same_rand_state=False
    )
    assert d_["a"] != d_["b"]

    # test class
    apply_to_features = ApplyToFeatures(["a", "b"], lambda x: x + np.random.rand())
    d_ = apply_to_features(d)
    assert d_["a"] == d_["b"]
