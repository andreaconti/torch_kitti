from torch.utils.data import Dataset
from ._data_elem import DataGroup, DataElem
from typing import Callable, Dict, List
import random


class _LoadPrev:
    def __init__(self, base: DataElem):
        self._previous = None
        self._base = base

    def reset(self):
        self._previous = None

    def __call__(self, elem: DataElem, load_previous=0, load_sequence=1) -> DataElem:
        if isinstance(load_previous, tuple):
            if self._previous is None:
                delta = random.randrange(load_previous[0], load_previous[1] + 1)
                self._previous = delta
            else:
                delta = self._previous
            if self._base.change_path(idx=elem.idx - delta).exists():
                elem.add_path(idx=elem.idx - delta)
            else:
                elem.add_path(idx=elem.idx)
        elif load_previous != 0:
            if self._previous is None:
                delta = load_previous
                self._previous = delta
            else:
                delta = self._previous
            if self._base.change_path(idx=elem.idx - delta).exists():
                elem.add_path(idx=elem.idx - delta)
            else:
                elem.add_path(idx=elem.idx)

        idx = elem.idx
        for delta in range(1, load_sequence):
            if self._base.change_path(idx=idx - delta).exists():
                elem.add_path(idx=idx - delta)
            else:
                elem.add_path(
                    idx=elem.idx if isinstance(elem.idx, int) else elem.idx[0]
                )

        return elem


def _identity(x):
    return x


class GenericDataset(Dataset):
    def __init__(
        self, elems: List[DataGroup], transform: Callable[[Dict], Dict] = _identity
    ):
        self.elems = elems
        self.transform = transform

    def __len__(self):
        return len(self.elems)

    def __getitem__(self, x):
        elems = self.elems[x]
        return self.transform(elems.data)
