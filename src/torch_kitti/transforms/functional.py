import random
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

__all__ = ["apply_to_features", "add_features"]


def apply_to_features(
    transform: Callable,
    x: Dict,
    features: Optional[List[str]] = None,
    same_rand_state=True,
) -> Dict:
    """
    Apply the same transformation to a subset of features of a dictionary.
    Even random transformations can be applied in the same way to each
    field cause the inner random state is the same for each application.

    Parameters
    ----------
    features: List[Any], optional
        A list of dictionary keys to apply `transform`, if not provided `transform`
        is applied to all fields
    transform: Callable
        Transformation to be applied
    same_rand_state: bool, default True
        If use the same random state for each application of `transform`.

    Returns
    -------
    The same input with specified `features` transformated by `transform`

    Example
    -------

    >>> d = {"a": np.random.randn(1, 3, 100, 100), "b": np.random.randn(1, 3, 100, 100)}
    >>> d = apply_to_features(lambda x: x + np.random.rand(), d, ["a", "b"])

    """

    if same_rand_state:

        # move randomness
        np.random.rand()
        random.random()
        torch.rand(1)

        # save state
        np_state = np.random.get_state()
        rd_state = random.getstate()
        tr_state = torch.random.get_rng_state()

    y = dict(**x)

    if features is None:
        features = list(x.keys())

    for feature in features:
        y[feature] = transform(x[feature])

        if same_rand_state:
            np.random.set_state(np_state)
            random.setstate(rd_state)
            torch.set_rng_state(tr_state)

    return y


def add_features(transform: Callable[[Dict], Dict], x: Dict) -> Dict:
    """
    Takes the input, passes it to a transformation and merge the original
    input with the result of the transformation, can be used to
    augment data in the batch.

    .. note::
        transform type and input type must match.

    Parameters
    ----------
    transform: Callable
        transformation applied
    x: Dict
        data to transform

    Example
    -------

    >>> add_features(lambda _: {'y': 1}, {'a': 2})
    {'a': 2, 'y': 1}
    """
    results = transform(x)
    if isinstance(results, dict) and isinstance(x, dict):
        return dict(x, **results)
    else:
        raise ValueError("dictionaries supported only")
