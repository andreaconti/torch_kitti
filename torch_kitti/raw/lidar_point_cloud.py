"""
Utilities to load and manipulate Lidar Point clouds
"""

import numpy as np

__all__ = ["load_lidar_point_cloud"]


def load_lidar_point_cloud(path: str, homologous: bool = True) -> np.ndarray:
    """
    Loads points from the binary file generated
    by a lidar and returns xyz homologous coordinates

    Parameters
    ----------
    path: str
        path to the file containing lidar points, in KITTI raw dataset
        these files are under 'velodyne_points/data' folder with .bin
        extension.
    homologous: bool, default True
        lidar points also contain also reflectance for each point, if
        `homologous` is True reflectance is substituted by '1.0' in
        order to have homologous coordinates.

    Returns
    -------
    ndarray
        returns a numpy ndarray of shape [N, 4] of N points with x y z 1
        coordinates if homologous is True otherwise 1 is substituted by
        reflectance

    Examples
    --------

    >>> load_lidar_points('0000000000.bin').shape
    (60540, 4)

    """
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    # last value should be reflectance. Is changed in homologous coordinates
    if homologous:
        points[:, 3] = 1.0
    return points
