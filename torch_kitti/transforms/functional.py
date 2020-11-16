import random
from typing import Callable, Dict, List, Optional

import numpy as np


def apply_to_features(
    features: Optional[List[str]], transform: Callable, x: Dict, same_rand_state=True
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
    >>> d = apply_to_features(["a", "b"], lambda x: x + np.random.rand(), d)

    """

    if same_rand_state:

        # move randomness
        np.random.rand()
        random.random()

        # save state
        np_state = np.random.get_state()
        rd_state = random.getstate()

    y = dict(**x)

    if features is None:
        features = list(x.keys())

    for feature in features:
        y[feature] = transform(x[feature])

        if same_rand_state:
            np.random.set_state(np_state)
            random.setstate(rd_state)

    return y
