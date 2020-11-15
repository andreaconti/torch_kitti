"""
Download utilities
"""

__all__ = ["calib_repos"]

import os
from pathlib import Path
from typing import List
from zipfile import ZipFile

import requests
from tqdm import tqdm


def calib_repos() -> List[str]:
    """
    Provides a plain list of urls pointing to all calib files of KITTI dataset.
    These are also available at the address here_.

    .. _here: http://www.cvlibs.net/datasets/kitti/raw_data.php
    """

    repos = [
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_calib.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip",
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_calib.zip",
    ]

    return repos


def download_file(url, save_path, chunk_size=1024, verbose=True):
    """
    Downloads a zip file from an `url` into a zip file in the
    provided `save_path`.
    """
    r = requests.get(url, stream=True)
    zip_name = url.split("/")[-1]

    content_length = int(r.headers["Content-Length"]) / 10 ** 6

    if verbose:
        bar = tqdm(total=content_length, unit="Mb", desc="download " + zip_name)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
            if verbose:
                bar.update(chunk_size / 10 ** 6)

    if verbose:
        bar.close()


def download_calib(root_path: str):
    """
    Downloads and scaffolds calibration files
    """

    Path(root_path).mkdir(exist_ok=True)

    for repo in calib_repos():
        calib_zip_path = os.path.join(root_path, "calib.zip")
        download_file(repo, calib_zip_path)
        with open(calib_zip_path, "rb") as f:
            ZipFile(f).extractall(root_path)
        os.remove(calib_zip_path)
