"""
Utilities to automate RAW data downloading from
:ref:`http://www.cvlibs.net/datasets/kitti/raw_data.php`
"""

import logging
import os
import pickle
import re
import shutil
from multiprocessing.pool import ThreadPool
from typing import List
from zipfile import ZipFile

from pkg_resources import resource_filename
from tqdm import tqdm

from .._download import download_calib, download_file

__all__ = [
    "repos",
    "check_drives",
    "download",
]

logger = logging.getLogger(__name__)


def repos() -> List[str]:
    """
    Provides a plain list of urls pointing to all syncronized+rectified
    .zip files of KITTI dataset.
    These are also available at the address here_.

    .. _here: http://www.cvlibs.net/datasets/kitti/raw_data.php
    """

    drives_path = resource_filename(
        "torch_kitti.raw.synced_rectified.resources", "drives.pkl"
    )
    with open(drives_path, "rb") as f:
        drives = pickle.load(f)

    drive = re.compile(r"[0-9]+_[0-9]+_[0-9]+_drive_[0-9]+")

    repos = [drive.findall(d)[0] + "/" + d + ".zip" for d in drives]

    prefix = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"

    def concat_prefix(repo: str) -> str:
        return prefix + repo

    return list(map(concat_prefix, repos))


def check_drives(root: str) -> bool:
    """
    Performs a check of all drives contained in the kitti root folders
    and prints a report
    """

    date_match = re.compile("[0-9]+_[0-9]+_[0-9]+")
    drive_match = re.compile(r"[0-9]+_[0-9]+_[0-9]+_drive_[0-9]+_sync")

    missing = []
    for url in repos():
        drive = drive_match.findall(url)[0]
        date = date_match.findall(url)[0]
        if not os.path.exists(os.path.join(root, date, drive)):
            missing.append(drive)

    if len(missing) != 0:
        logging.error("missing drives: " + "\n".join("- " + drive for drive in missing))
        return False
    else:
        return True


def download(root_path: str, threads=6):
    """
    Provides a simple way to download and scaffold the whole kitti synced+rectified
    raw data in a folder from :ref:`http://www.cvlibs.net/datasets/kitti/raw_data.php`.
    The whole KITTI dataset contains about 170Gb.
    """

    date_match = re.compile("[0-9]+_[0-9]+_[0-9]+")
    drive_match = re.compile(r"[0-9]+_[0-9]+_[0-9]+_drive_[0-9]+_sync")

    # filter not existing repos
    all_repos = [
        url
        for url in repos()
        if not os.path.exists(
            os.path.join(
                root_path, date_match.findall(url)[0], drive_match.findall(url)[0]
            )
        )
    ]

    # chunk generator
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    bar = tqdm(total=len(all_repos), unit="zips", desc="download repos")

    def download_unzip(url):
        # params
        date = date_match.findall(url)[0]
        drive = drive_match.findall(url)[0]

        # create date folder if not exist
        os.makedirs(os.path.join(root_path, date), exist_ok=True)

        # download
        download_file(url, os.path.join(root_path, date, drive + ".zip"))

        # unzip
        with ZipFile(os.path.join(root_path, date, drive + ".zip"), "r") as zip_ref:
            zip_ref.extractall(os.path.join(root_path, date, drive + "_tmp"))

        # move folders
        os.rename(
            os.path.join(root_path, date, drive + "_tmp", date, drive),
            os.path.join(root_path, date, drive),
        )

        # remove old folder and zip file
        shutil.rmtree(os.path.join(root_path, date, drive + "_tmp"))
        os.remove(os.path.join(root_path, date, drive + ".zip"))

        # update bar
        bar.update(1)

    try:
        # download calibration files
        download_calib(root_path)

        # download drives
        for urls in chunks(repos(), threads):
            pool = ThreadPool(threads)
            pool.map(download_unzip, urls)
    except (KeyboardInterrupt, Exception) as e:
        for url in urls:
            # params
            date = date_match.findall(url)[0]
            drive = drive_match.findall(url)[0]

            # clean drive
            drive_path = os.path.join(root_path, date, drive)
            logger.info("removing " + drive_path)
            shutil.rmtree(drive_path, ignore_errors=True)
            if os.path.exists(drive_path + ".zip"):
                os.remove(drive_path + ".zip")
            shutil.rmtree(drive_path + "_tmp", ignore_errors=True)

        # reraise error
        raise e
    finally:
        bar.close()
