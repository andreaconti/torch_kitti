import numpy as np
import pytest
import torch

from torch_kitti.metrics import functional as F


def test_rmse():
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 3.0, 4.0])

    assert F.rmse(y_pred, y_pred) == 0
    assert F.rmse(y_pred, y_true) == 1


def test_mse():
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 3.0, 4.0])

    assert F.mse(y_pred, y_pred) == 0
    assert F.mse(y_pred, y_true) == 1


def test_irmse():
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 3.0, 4.0])

    assert F.irmse(y_pred, y_pred) == 0
    assert torch.isclose(F.irmse(y_pred, y_true), torch.tensor(0.30807))


def test_mae():
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 3.0, 4.0])

    assert F.mae(y_pred, y_pred) == 0
    assert F.mae(y_pred, y_true) == 1


def test_imae():
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 3.0, 4.0])

    assert F.imae(y_pred, y_pred) == 0
    assert F.imae(y_pred, y_true) == 0.25


def test_sq_rel_error():
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 3.0, 4.0])

    assert F.sq_rel_error(y_pred, y_pred) == 0
    assert torch.isclose(F.sq_rel_error(y_pred, y_true), torch.tensor(0.36111))


def test_silog():
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 3.0, 4.0])

    assert F.silog(y_pred, y_pred) == 0
    assert np.isclose(F.silog(y_pred, y_true).item(), 0.029003977)


@pytest.mark.parametrize(
    "y_pred, y_true",
    [(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 3.0, 4.0]))],
)
@pytest.mark.parametrize(
    "threshold, expected",
    [
        (1.25, 0.0),
        (1.25 ** 2, 0.6666666),
        (1.25 ** 3, 0.6666666),
    ],
)
def test_ratio_threshold(y_pred, y_true, threshold, expected):
    assert np.isclose(F.ratio_threshold(y_pred, y_true, threshold).item(), expected)
