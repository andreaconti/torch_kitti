"""
Utilities to load and manipulate Lidar Point clouds
"""

import numpy as np

__all__ = ["load_lidar_point_cloud"]


def load_lidar_point_cloud(path: str, projective: bool = True) -> np.ndarray:
    """
    Loads points from the binary file generated
    by a lidar and returns xyz projective coordinates

    Parameters
    ----------
    path: str
        path to the file containing lidar points, in KITTI raw dataset
        these files are under 'velodyne_points/data' folder with .bin
        extension.
    projective: bool, default True
        lidar points also contain also reflectance for each point, if
        `projective` is True reflectance is substituted by '1.0' in
        order to have projective coordinates.

    Returns
    -------
    ndarray
        returns a numpy ndarray of shape [N, 4] of N points with x y z 1
        coordinates if projective is True otherwise 1 is substituted by
        reflectance

    Examples
    --------

    >>> load_lidar_points('0000000000.bin').shape
    (60540, 4)

    """
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    # last value should be reflectance. Is changed in projective coordinates
    if projective:
        points[:, 3] = 1.0
    return points
