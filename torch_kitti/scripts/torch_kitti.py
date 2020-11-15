"""
Main program for torch kitti
"""

from argparse import ArgumentParser

from torch_kitti.depth_completion import download as depth_completion_download
from torch_kitti.depth_prediction import download as depth_prediction_download
from torch_kitti.raw.synced_rectified import download as sync_rect_download

# dowload datasets

download_funcs = {
    "sync_rectified": sync_rect_download,
    "depth_completion": depth_completion_download,
    "depth_prediction": depth_prediction_download,
}


def main():

    # parser
    parser = ArgumentParser("Torch Kitti")
    subparsers = parser.add_subparsers(help="sub-command help")

    # download subparser
    download_parser = subparsers.add_parser("download", help="scaffolding utilities")
    download_parser.add_argument(
        "name",
        type=str,
        choices=download_funcs.keys(),
        help="name of the dataset to download",
    )
    download_parser.add_argument("path", type=str, help="where scaffold the dataset")
    download_parser.set_defaults(func=lambda args: download_funcs[args.name](args.path))

    # run
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
