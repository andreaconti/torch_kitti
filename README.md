# Pytorch KITTI

This project aims to provide a simple yet effective way to scaffold and load the [KITTI Vision Banchmark Dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) providing **Datasets**, a simple way to **download** them, **metrics** and **transformations**.

## Installation

To install `torch-kitti`

```bash
$ pip install torch-kitti
```

## Scaffolding datasets

To manually download the datasets `torch-kitti` command line utility comes in handy:

```bash
$ torch_kitti download --help
usage: Torch Kitti download [-h]
                            {sync_rectified,depth_completion,depth_prediction}
                            path

positional arguments:
  {sync_rectified,depth_completion,depth_prediction}
                        name of the dataset to download
  path                  where scaffold the dataset

optional arguments:
  -h, --help            show this help message and exit
```

Actually available datasets are:

- KITTI Depth Completion Dataset
- KITTI Depth Prediction Dataset
- KITTI Raw Sync+Rect Dataset

## Loading Datasets

All datasets return dictionaries, utilities to manipulate them can be found in `torch_kitti.transforms` module. Often each dataset provides options to include optional fields, for instance `KittiDepthCompletionDataset` usually provides simply the `img`, its sparse depth groundtruth `gt` and the sparse lidar hints `lidar` but using `load_stereo=True` stereo images will be included for each example.

```python
from torchvision.transforms import Compose, RandomCrop, ToTensor

from torch_kitti.depth_completion import KittiDepthCompletionDataset
from torch_kitti.transforms import ApplyToFeatures

transform = ApplyToFeatures(
    Compose(
        [
            ToTensor(),
            RandomCrop([256, 512]),
        ]
    ),
    features=["img", "gt", "lidar"],
)

ds = KittiDepthCompletionDataset(
    "kitti_raw_sync_rect_root",
    "kitti_depth_completion_root",
    load_stereo=False,
    transform=transform,
    download=True,  # download if not found
)
```

## Develop

Download from kitti and `cd` in the folder then prepare a virtual environment (1), install `dev` and `doc` dependencies (2) and `pre-commit` (3).

```bash
$ git clone https://github.com/andreaconti/torch_kitti.git
$ cd torch_kitti
$ python3 -m virtualenv .venv && source .venv/bin/activate  # (1)
$ pip install .[dev, doc] # (2)
$ pre-commit install  # (3)
$ python3 setup.py develop
$ pytest
```

Tests use some environment variables to locate each dataset on the file system and perform specific tests on it. If they are not found tests are skipped.

* KITTI_SYNC_RECT_ROOT: root of the kitti sync rect dataset
* KITTI_DEPTH_COMPLETION_ROOT: root of the kitti depth completion dataset
