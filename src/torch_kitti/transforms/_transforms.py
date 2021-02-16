from typing import Any, Callable, Dict, List, Optional

from .functional import add_features, apply_to_features

__all__ = ["ApplyToFeatures", "AddFeatures"]


class ApplyToFeatures:
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
    >>> d = ApplyToFeatures(lambda x: x + np.random.rand(), ["a", "b"])(d)

    """

    def __init__(
        self,
        transform: Callable[[Any], Any],
        features: Optional[List[Any]] = None,
        same_rand_state: bool = True,
    ):
        self.features = features
        self.transform = transform
        self.same_rand_state = same_rand_state

    def __call__(self, x: Dict) -> Dict:
        return apply_to_features(self.transform, x, self.features, self.same_rand_state)


class AddFeatures:
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

    def __init__(self, transform: Callable[[Dict], Dict]):
        self.transform = transform

    def __call__(self, x: Dict) -> Dict:
        return add_features(self.transform, x)
