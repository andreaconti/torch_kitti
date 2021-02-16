"""
Utilities to download, scaffold, load and manipulate KITTI Raw Dataset
"""

__all__ = ["download", "check_drives"]

from ._download import check_drives, download
