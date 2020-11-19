# Pytorch KITTI


[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Python version](https://img.shields.io/badge/python-3.6|3.7|3.8-green.svg)
[![PyPI version](https://badge.fury.io/py/torch-kitti.svg)](https://badge.fury.io/py/torch-kitti)
![License](https://img.shields.io/pypi/l/torch-kitti)

This project aims to provide a simple yet effective way to scaffold and load the [KITTI Vision Banchmark Dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) providing

- **Datasets**: Pytorch datasets to load each dataset

- **Scaffolding**: to download the datasets

- **Metrics**: common metrics used for each dataset

- **Transformations**: utilities to manipulate samples

## Installation

To install `torch-kitti`

```bash
$ pip install torch-kitti
```

## Scaffolding datasets

To manually download the datasets the `torch-kitti` command line utility comes in handy:

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

All datasets return dictionaries, utilities to manipulate them can be found in the `torch_kitti.transforms` module. Often each dataset provides options to include optional fields, for instance `KittiDepthCompletionDataset` usually provides simply the `img`, its sparse depth groundtruth `gt` and the sparse lidar hints `lidar` but using `load_stereo=True` stereo images will be included for each example.

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

## Contributing

### Developing setup

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

Feel free to open an issue on [GitHub](https://github.com/andreaconti/torch_kitti/issues), fork the [repository](https://github.com/andreaconti/torch_kitti) and submit a pull request to solve bugs, improve docs, add datasets and features. All new feature must be tested.



## Disclaimer on KITTI Vision Benchmark Suite

This library is an utility that downloads and prepares the dataset. The KITTI Vision Benchmark Suite is not hosted by this project nor it's claimed that you have license to use the dataset, it is your responsibility to determine whether you have permission to use this dataset under its license. You can find more details [here](http://www.cvlibs.net/datasets/kitti/).
