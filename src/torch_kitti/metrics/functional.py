from typing import Union

import torch

# DEPTH COMPLETION METRICS

__all__ = [
    "rmse",
    "mse",
    "irmse",
    "mae",
    "imae",
    "sq_rel_error",
    "abs_rel_error",
    "silog",
    "ratio_threshold",
]


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    computes the root mean squared error
    """
    return torch.sqrt(torch.mean(torch.square(y_pred - y_true)))


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    computes the mean squared error
    """
    return torch.mean(torch.square(y_pred - y_true))


def irmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the inverse root mean squared error

    .. math::

        \sqrt{\frac{1}{|V|}\sum_{v \in V}|\frac{1}{d_v^{gt}} - \frac{1}{d_v^{pred}}|^2}
    """
    return torch.sqrt(torch.mean(torch.square(1 / y_true - 1 / y_pred)))


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    computes the mean absolute error
    """
    return torch.mean(torch.abs(y_pred - y_true))


def imae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    r"""
    Compute inverse mean absolute error

    .. math::

        \frac{1}{|V|}\sum_{v \in V}|\frac{1}{d_v^{gt}} - \frac{1}{d_v^{pred}}|
    """
    return torch.mean(torch.abs(1 / y_true - 1 / y_pred))


# DEPTH PREDICTION METRICS


def sq_rel_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the relative L2 error with respect to
    y_true

    .. math::

        \frac{(y - \hat{y})^2}{\hat{y}}
    """
    return torch.mean(torch.square(y_pred - y_true) / y_true)


def abs_rel_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the relative L1 error with respect to
    y_true

    .. math::

        \frac{|y - \hat{y}|}{\hat{y}}
    """
    return torch.mean(torch.abs(y_pred - y_true) / y_true)


def silog(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the scale invariant logarithmig error as states here_

    .. _here: https://arxiv.org/abs/1406.2283
    """

    return torch.mean(
        torch.square(torch.log(y_pred) - torch.log(y_true))
    ) - torch.square(torch.mean(torch.log(y_pred) - torch.log(y_true)))


def ratio_threshold(
    threshold: Union[torch.Tensor, float], y_pred: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    r"""
    Computes the percentage of values whose maximum between thre ratio and the inverse
    ratio with respect to the `y_true` is lower than the `threshold`.
    """
    result = torch.max(y_true / y_pred, y_pred / y_true)
    result = (result < threshold).to(torch.float32).mean()
    return result
