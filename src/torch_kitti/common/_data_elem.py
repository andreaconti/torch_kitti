from multiprocessing.sharedctypes import Value
from typing import Dict, Iterable, List, Literal, Optional, Union
import numpy as np
from torch_kitti.raw.calibration import CamCalib
from torch_kitti.raw.inertial_measurement_unit import IMUData
from pathlib import Path
import re
import os
from PIL import Image

from torch_kitti.raw.lidar_point_cloud import load_lidar_point_cloud

StrOrPath = Union[str, Path]

__all__ = ["DataElem", "DataGroup"]


class DataElem:
    """
    It contains the logic to load all the useful components from the KITTI Dataset
    abstracting from the source and providing useful utilities and metadata
    """

    def __init__(
        self,
        name: str,
        type: Literal["image", "depth", "pcd", "calib", "imu", "rt", "intrinsics"],
        path: StrOrPath,
        cam: Optional[Union[int, Literal[False]]] = None,
        drive: Optional[str] = None,
        idx: Optional[int] = None,
        **kwargs,
    ):
        # checks
        assert type in [
            "image",
            "depth",
            "pcd",
            "calib",
            "imu",
            "rt",
            "intrinsics",
        ], ValueError(f"type {type} not supported")

        # fields
        self.name = name
        self.type = type
        self._path = [Path(path)]
        self.cam = cam
        self.drive = drive
        self._idx = [idx]
        self._data = None
        self.opts = kwargs

        # try to infer cam, drive, idx
        path_str = str(self.path.as_posix())
        if drive is None:
            match = re.search("[0-9]+_[0-9]+_[0-9]+_drive_[0-9]+_sync", path_str)
            if not match:
                raise RuntimeError(
                    f"can't infer drive from path {self.path}, explicit provide it"
                )
            else:
                self.drive = match.group()
        match = re.search("[0-9]+_[0-9]+_[0-9]+", self.drive)
        if not match:
            raise RuntimeError(f"can't extract date from {self.drive}")
        else:
            self.date = match.group()

        if idx is None:
            match = re.search("[0-9]{10}", path_str)
            if not match:
                raise RuntimeError(
                    f"can't infer idx from path {self.path}, explicit provide it"
                )
            else:
                self._idx = [int(match.group())]
        if cam is None:
            match = re.search("image_[0-9]{2}", path_str)
            if not match:
                raise RuntimeError(
                    f"can't infer cam from path {self.path}, explicit provide it"
                )
            else:
                self.cam = int(match.group()[-2:])

        self._load_funcs = {
            "image": self._load_img,
            "depth": self._load_depth,
            "pcd": self._load_pcd,
            "calib": self._load_calib,
            "imu": self._load_imu,
            "rt": self._load_rt,
            "intrinsics": self._load_intrinsics,
        }

    # functions to load specific file types
    def _load_img(self) -> np.ndarray:
        if len(self._path) > 1:
            return np.stack([np.array(Image.open(p)) for p in self._path])
        else:
            return np.array(Image.open(self._path[0]))

    def _load_pcd(self) -> Union[List[np.ndarray], np.ndarray]:
        pcds = [
            load_lidar_point_cloud(
                p, self.opts.get("pcd_format", "projective") == "projective"
            )
            for p in self._path
        ]
        if len(pcds) == 1:
            pcds = pcds[0]
        return pcds

    def _load_single_rt(self, path: Path) -> np.ndarray:
        dict_values: Dict[str, np.ndarray] = dict()
        with open(path, "rt") as f:
            for line in f.readlines():
                name, vector = line.split(":", maxsplit=1)
                if name == "calib_time":
                    continue
                vector = np.array(list(map(float, vector.strip().split(" "))))
                dict_values[name] = vector

        R = dict_values["R"].reshape(3, 3)
        T = dict_values["T"]
        rt = np.eye(4)
        rt[:3, :3] = R
        rt[:3, 3] = T

        return rt

    def _load_rt(self) -> np.ndarray:
        if len(self._path) > 1:
            return np.stack([self._load_single_rt(p) for p in self._path])
        else:
            return self._load_single_rt(self._path[0])

    def _load_single_depth(self, path) -> np.ndarray:
        depth = np.array(Image.open(path)).astype(np.float32) / 256.0
        depth = np.expand_dims(depth, axis=-1)
        return depth

    def _load_depth(self) -> np.ndarray:
        if len(self._path) > 1:
            return np.stack([self._load_single_depth(p) for p in self._path])
        else:
            return self._load_single_depth(self._path[0])

    def _load_calib(self) -> Union[List[CamCalib], CamCalib]:
        calibs = [CamCalib.open(self.cam, p) for p in self._path]
        return calibs if len(calibs) > 1 else calibs[0]

    def _load_intrinsics(self) -> np.ndarray:
        if len(self._path) > 1:
            return np.stack(
                [
                    CamCalib.open(self.cam, p).projection_matrix[:3, :3]
                    for p in self._path
                ]
            )
        else:
            return CamCalib.open(self.cam, self._path[0]).projection_matrix[:3, :3]

    def _load_imu(self) -> Union[List[IMUData], IMUData]:
        imu = [IMUData.open(p) for p in self._path]
        return imu if len(imu) > 1 else imu[0]

    # properties

    @property
    def data(self):
        if self._data is None:
            self._data = self._load_funcs[self.type]()
        return self._data

    @property
    def idx(self) -> Union[List[int], int]:
        if len(self._idx) > 1:
            return self._idx
        else:
            return self._idx[-1]

    @property
    def path(self) -> Union[List[Path], Path]:
        if len(self._path) > 1:
            return self._path
        else:
            return self._path[-1]

    def exists(self) -> bool:
        return all(path.exists() for path in self._path)

    def change_path(self, idx: int) -> Path:
        curr_path = str(self._path[-1].as_posix())
        curr_idx = self._idx[-1]
        return Path(curr_path.replace(f"{curr_idx:0>10}", f"{idx:0>10}"))

    def add_path(
        self,
        idx: Optional[int] = None,
        path: Optional[StrOrPath] = None,
    ):

        if path:
            path = Path(path)
            assert not idx, "if path is provided idx not allowed"
            match = re.search("[0-9]{10}", str(path.as_posix()))
            if not match:
                raise RuntimeError(
                    f"can't infer idx from path {path}, explicit provide it"
                )
            else:
                self._idx = [int(match.group())] + self._idx
            self._path = [path] + self._path
        else:
            assert idx is not None, "if path not provided, idx required"
            self._path = [self.change_path(idx)] + self._path
            self._idx = [idx] + self._idx

        self._data = None

    def remove_path(self, idx: int):
        idx = self._idx.index(idx)
        self._idx.pop(idx)
        self._path.pop(idx)

    def __repr__(self):
        return f"DataElem(name={self.name}, type={self.type}, drive={self.drive}, cam={self.cam}, idx={self.idx})"

    def __eq__(self, other):
        return (
            self.type == other.type
            and self.drive == other.drive
            and self.cam == other.cam
            and self.idx == other.idx
        )


class DataGroup:
    """
    It groups together multiple data sources and provides some common metadata
    """

    def __init__(self, elems: List[DataElem]):
        self.drive = elems[0].drive
        self.idx = elems[0].idx
        self.cam = set()
        for elem in elems:
            if elem.drive != self.drive:
                raise ValueError("all elems must have the same drive")
            if elem.idx != self.idx:
                raise ValueError("all elems must have the same idx")
            if elem.cam is not False:
                self.cam.add(elem.cam)

        if len(self.cam) == 1:
            self.cam = list(self.cam)[0]
        elif len(self.cam) == 0:
            self.cam = False

        self._elems = elems

    @property
    def fields(self) -> List[str]:
        return sorted([e.name for e in self._elems])

    @property
    def elems(self) -> List[DataElem]:
        return self._elems

    @property
    def data(self) -> Dict:
        output = {elem.name: elem.data for elem in self._elems}
        return output

    def remove(self, name: str):
        self._elems = [elem for elem in self._elems if elem.name != name]

    def add(self, elem: DataElem):
        assert elem.drive == self.drive, "all elems must have the same drive"
        assert elem.idx == self.idx, "all elems must have the same idx"
        self._elems.append(elem)

    def __repr__(self):
        return f"DataGroup(drive={self.drive}, idx={self.idx}, fields={self.fields}, cam={self.cam})"
