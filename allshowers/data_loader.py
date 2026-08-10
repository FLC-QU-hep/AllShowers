import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

import numpy as np
import torch

__all__ = ["DataSet", "DataLoader", "DictDataSet", "MMapDictDataSet", "ModelInputDict"]


class ModelInputDict(TypedDict):
    x: torch.Tensor
    cond: torch.Tensor
    num_points: torch.Tensor
    layer: torch.Tensor
    mask: torch.Tensor
    label: torch.Tensor
    noise: torch.Tensor | None


class DataSet(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int | list[int] | torch.Tensor) -> ModelInputDict:
        pass


class DataLoader(Iterable[ModelInputDict]):
    data_set: DataSet
    batch_size: int
    drop_last: bool
    shuffle: bool
    max_batch: int

    class __BatchIterator(Iterator[ModelInputDict]):
        def __init__(self, data_loader: "DataLoader", index: torch.Tensor) -> None:
            self.data_loader = data_loader
            self.batch = 0
            self.index = index
            self._prefetch_future = None
            if data_loader.prefetch:
                self._executor = ThreadPoolExecutor(max_workers=1)
                self._prefetch_next()

        def _load_batch(self, batch_num: int) -> ModelInputDict:
            first = batch_num * self.data_loader.batch_size
            last = min(first + self.data_loader.batch_size, len(self.index))
            idx = self.index[first:last]
            return self.data_loader.data_set[idx]

        def _prefetch_next(self) -> None:
            if self.batch < self.data_loader.max_batch:
                self._prefetch_future = self._executor.submit(
                    self._load_batch, self.batch
                )

        def __next__(self) -> ModelInputDict:
            if self.batch >= self.data_loader.max_batch:
                raise StopIteration
            if self._prefetch_future is not None:
                result = self._prefetch_future.result()
                self.batch += 1
                self._prefetch_next()
                return result
            else:
                first = self.batch * self.data_loader.batch_size
                last = min(first + self.data_loader.batch_size, len(self.index))
                idx = self.index[first:last]
                self.batch += 1
                return self.data_loader.data_set[idx]

    def __init__(
        self,
        data_set: DataSet,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        batch_shuffle: bool = False,
        prefetch: bool = False,
    ) -> None:
        self.data_set = data_set
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_shuffle = batch_shuffle
        self.prefetch = prefetch

        if self.drop_last:
            self.max_batch = len(self.data_set) // self.batch_size
        else:
            self.max_batch = (
                len(self.data_set) + self.batch_size - 1
            ) // self.batch_size

    def __len__(self) -> int:
        return self.max_batch

    def __iter__(self) -> Iterator[ModelInputDict]:
        n = len(self.data_set)
        if self.shuffle:
            if self.batch_shuffle:
                # Shuffle batch order but keep indices within each batch contiguous
                # Much faster for I/O-bound lazy datasets
                batch_starts = list(range(0, n, self.batch_size))
                perm = torch.randperm(len(batch_starts))
                index = torch.cat(
                    [
                        torch.arange(
                            batch_starts[i], min(batch_starts[i] + self.batch_size, n)
                        )
                        for i in perm
                    ]
                )
            else:
                index = torch.randperm(n)
        else:
            index = torch.arange(n)

        return self.__BatchIterator(self, index)


class DictDataSet(DataSet):
    def __init__(self, data: ModelInputDict) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data["x"])

    def __getitem__(self, index: int | list[int] | torch.Tensor) -> ModelInputDict:
        data = {}
        for key, value in self.data.items():
            if type(value) is torch.Tensor:
                data[key] = value[index].clone().detach()
            else:
                data[key] = None
        result = ModelInputDict(**data)
        return result


_CACHE_KEYS = ["x", "cond", "num_points", "layer", "mask", "label", "noise"]


def save_cache(data: ModelInputDict, cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for key in _CACHE_KEYS:
        val = data[key]
        if val is not None:
            np.save(os.path.join(cache_dir, f"{key}.npy"), val.numpy())
    print(f"Saved preprocessed cache to {cache_dir}")


class MMapDictDataSet(DataSet):
    def __init__(self, cache_dir: str, start: int = 0, stop: int | None = None) -> None:
        self._tensors: dict[str, torch.Tensor | None] = {}
        for key in _CACHE_KEYS:
            path = os.path.join(cache_dir, f"{key}.npy")
            if os.path.exists(path):
                arr = np.load(path, mmap_mode="r")[start:stop]
                if key == "layer":
                    arr = np.array(arr, dtype=np.int8)
                else:
                    arr = np.array(arr)
                self._tensors[key] = torch.from_numpy(arr)
                print(
                    f"  loaded {key}: {self._tensors[key].shape} "
                    f"({self._tensors[key].nbytes / 1e9:.1f} GB)"
                )
            else:
                self._tensors[key] = None
        self._len = len(self._tensors["x"])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int | list[int] | torch.Tensor) -> ModelInputDict:
        data = {}
        for key in _CACHE_KEYS:
            val = self._tensors[key]
            if val is not None:
                t = val[index].clone().detach()
                if key == "layer":
                    t = t.to(torch.int64)
                data[key] = t
            else:
                data[key] = None
        return ModelInputDict(**data)
