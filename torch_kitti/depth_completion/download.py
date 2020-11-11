import os
import shutil
from zipfile import ZipFile

import requests
from tqdm import tqdm

__all__ = ["download", "folders_check"]


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


def download(root_path: str, verbose: bool = True):
    """
    Downloads and scaffold depth completion dataset in `root_path`
    """

    # urls
    data_depth_selection_url = (
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip"
    )
    data_depth_velodyne_url = (
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip"
    )
    data_depth_annotated_url = (
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
    )

    # download of zips
    download_file(
        data_depth_selection_url,
        os.path.join(root_path, "data_depth_selection.zip"),
    )
    download_file(
        data_depth_velodyne_url,
        os.path.join(root_path, "data_depth_velodyne.zip"),
    )
    download_file(
        data_depth_annotated_url,
        os.path.join(root_path, "data_depth_annotated.zip"),
    )

    # unzip and remove zips
    if verbose:
        print("unzipping...")

    with ZipFile(os.path.join(root_path, "data_depth_selection.zip"), "r") as zip_ref:
        zip_ref.extractall(root_path)
        os.rename(
            os.path.join(
                root_path, "depth_selection", "test_depth_completion_anonymous"
            ),
            os.path.join(root_path, "test_depth_completion_anonymous"),
        )
        os.rename(
            os.path.join(
                root_path, "depth_selection", "test_depth_prediction_anonymous"
            ),
            os.path.join(root_path, "test_depth_prediction_anonymous"),
        )
        os.rename(
            os.path.join(root_path, "depth_selection", "val_selection_cropped"),
            os.path.join(root_path, "val_selection_cropped"),
        )
        os.rmdir(os.path.join(root_path, "depth_selection"))
    with ZipFile(os.path.join(root_path, "data_depth_velodyne.zip"), "r") as zip_ref:
        zip_ref.extractall(root_path)
    with ZipFile(os.path.join(root_path, "data_depth_annotated.zip"), "r") as zip_ref:
        zip_ref.extractall(root_path)

    # remove zip files
    shutil.rmtree(os.path.join(root_path, "data_depth_selection.zip"))
    shutil.rmtree(os.path.join(root_path, "data_depth_completion.zip"))
    shutil.rmtree(os.path.join(root_path, "data_depth_velodyne.zip"))

    if verbose:
        print("done.")


def folders_check(root_path: str):
    """
    Performs some simple checks about folders structure for depth completion dataset and
    prints errors
    """

    ok = True

    for folder in [
        "test_depth_completion_anonymous",
        "test_depth_prediction_anonymous",
        "val_selection_cropped",
    ]:
        if not os.path.exists(os.path.join(root_path, folder)):
            print(f"missing data_depth_selection.zip: folder {folder} not found")
            print()
            ok = False

    # check of data_depth_completion.zip and data_depth_velodyne.zip
    def check_gt_velodyne(folder):
        num_groundtruth, num_velodyne = 0, 0
        for dirpath, dirnames, filenames in os.walk(os.path.join(root_path, folder)):
            if "groundtruth" in dirnames:
                num_groundtruth += 1
            if "velodyne_raw" in dirnames:
                num_velodyne += 1
        return num_groundtruth, num_velodyne

    gt_train, velodyne_train = check_gt_velodyne("train")
    if gt_train != 138 and velodyne_train != 138:
        ok = False
        print("something went wrong with data_depth_completion.zip and", end=" ")
        print("data_depth_velodyne.zip in val folder")

        print(f"found {gt_train} groundtruth folders and", end=" ")
        print(f"{velodyne_train} raw_lidar folders but they should be 138..")
        print()

    gt_val, velodyne_val = check_gt_velodyne("val")
    if gt_val != 13 and velodyne_val != 13:
        ok = False
        print("something went wrong with data_depth_completion.zip and", end=" ")
        print("data_depth_velodyne.zip in val folder")

        print(f"found {gt_train} groundtruth folders and {velodyne_train}", end=" ")
        print("raw_lidar folders but the should be 13..")
        print()

    if ok:
        print("all good what ends well")
