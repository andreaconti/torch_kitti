from typing import Any, Callable, Dict, List, Optional

from .functional import apply_to_features


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
    >>> d = ApplyToFeatures(["a", "b"], lambda x: x + np.random.rand())(d)

    """

    def __init__(
        self,
        features: Optional[List[Any]],
        transform: Callable[[Any], Any],
        same_rand_state: bool = True,
    ):
        self.features = features
        self.transform = transform
        self.same_rand_state = same_rand_state

    def __call__(self, x: Dict) -> Dict:
        return apply_to_features(self.features, self.transform, x, self.same_rand_state)
