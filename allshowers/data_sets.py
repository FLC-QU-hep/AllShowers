import os
import time
import warnings
from typing import TypedDict

import h5py
import numpy as np
import showerdata
import torch
from torch import Tensor

from allshowers.data_loader import (
    DataLoader,
    DictDataSet,
    MMapDictDataSet,
    ModelInputDict,
)
from allshowers.preprocessing import Identity, Transformation, compose

__all__ = ["create_label_list", "to_label_tensor", "get_data_loaders"]


class ShowerDict(TypedDict):
    shower: Tensor
    energy: Tensor
    direction: Tensor
    pdg: Tensor
    noise: Tensor | None
    sampling_frac: Tensor | None
    nlayers: Tensor | None


def batched_histogram(
    data: torch.Tensor, mask: torch.Tensor, num_bins: int = -1
) -> torch.Tensor:
    if num_bins < 0:
        num_bins = int(torch.max(data[mask]).item()) + 1
    histograms = torch.zeros(
        size=(data.shape[0], num_bins), dtype=torch.int32, device=data.device
    )
    ones = torch.zeros(size=data.shape, dtype=histograms.dtype, device=data.device)
    ones[mask] = 1
    histograms.scatter_add_(1, data, ones)
    return histograms


@torch.no_grad()
def initialise_trafos(
    energies: Tensor,
    showers: Tensor,
    mask: Tensor,
    samples_energy_trafo: Transformation,
    samples_coordinate_trafo: Transformation,
    cond_trafo: Transformation,
    sampling_frac_trafo: Transformation | None = None,
    sampling_frac: Tensor | None = None,
    nlayers_trafo: Transformation | None = None,
    nlayers: Tensor | None = None,
    *,
    fit_stop: int | None = None,
    trafos_file: str = "",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    if trafos_file is None and world_size > 1:
        raise ValueError(
            "If using distributed training, a trafos_file must be provided to save and load the transformations."
        )
    if world_size > 1:
        torch.distributed.barrier(device_ids=[local_rank])
    if rank != 0:
        torch.distributed.barrier(device_ids=[local_rank])
    if os.path.isfile(trafos_file):
        if world_size > 1 and rank == 0:
            torch.distributed.barrier(device_ids=[local_rank])
        parameters = torch.load(trafos_file, weights_only=True)
        samples_energy_trafo.load_state_dict(parameters["samples_energy_trafo"])
        samples_coordinate_trafo.load_state_dict(parameters["samples_coordinate_trafo"])
        cond_trafo.load_state_dict(parameters["cond_trafo"])
        if sampling_frac_trafo is not None and "sampling_frac_trafo" in parameters:
            sampling_frac_trafo.load_state_dict(parameters["sampling_frac_trafo"])
        if nlayers_trafo is not None and "nlayers_trafo" in parameters:
            nlayers_trafo.load_state_dict(parameters["nlayers_trafo"])
        print(f"[rank {rank}] Loaded transformations from {trafos_file}")
    else:
        if rank != 0:
            raise RuntimeError(
                "Initialization of transformations is only allowed for rank 0"
            )
        n = fit_stop if fit_stop is not None else 100_000
        energies_l = energies[:n]
        showers_l = showers[:n]
        mask_l = mask[:n]
        cond_trafo.fit(energies_l)
        samples_coordinate_trafo.fit(showers_l[:, :, :2], mask_l)
        samples_energy_trafo.fit(showers_l[:, :, 3], mask_l.squeeze())
        if sampling_frac_trafo is not None and sampling_frac is not None:
            sampling_frac_l = sampling_frac[:n]
            sampling_frac_trafo.fit(sampling_frac_l)
        if nlayers_trafo is not None and nlayers is not None:
            nlayers_l = nlayers[:n]
            nlayers_trafo.fit(nlayers_l)
        if trafos_file:
            parameters = {
                "samples_energy_trafo": samples_energy_trafo.state_dict(),
                "samples_coordinate_trafo": samples_coordinate_trafo.state_dict(),
                "cond_trafo": cond_trafo.state_dict(),
            }
            if sampling_frac_trafo is not None:
                parameters["sampling_frac_trafo"] = sampling_frac_trafo.state_dict()
            if nlayers_trafo is not None:
                parameters["nlayers_trafo"] = nlayers_trafo.state_dict()
            torch.save(parameters, trafos_file)
            print(f"[rank {rank}] Saved transformations to {trafos_file}")
        if world_size > 1:
            time.sleep(5)  # make sure file is on network drive
            torch.distributed.barrier(device_ids=[local_rank])


def load_data(
    path: str,
    *,
    start: int = 0,
    stop: int | None = None,
    return_noise: bool = False,
    noise_path: str | None = None,
    return_sampling_frac: bool = False,
    return_nlayers: bool = False,
    max_num_points: int | None = None,
) -> ShowerDict:
    showers = showerdata.load(
        path,
        start,
        stop,
        max_points=max_num_points,
    )
    if return_noise:
        _noise_file = noise_path if noise_path is not None else path
        noise, _ = showerdata.load_target(_noise_file, "target", start=start, stop=stop)
    else:
        noise = None

    # Load sampling fraction if requested
    sampling_frac = None
    if return_sampling_frac:
        with h5py.File(path, "r") as f:
            if "sampling_fraction" in f:
                sf_data = f["sampling_fraction"][start:stop]
                # Ensure shape is (n_showers, 1)
                if sf_data.ndim == 1:
                    sf_data = sf_data[:, np.newaxis]
                sampling_frac = torch.from_numpy(sf_data.astype(np.float32))
            else:
                warnings.warn(
                    f"sampling_fraction not found in {path}, returning None",
                    UserWarning,
                )

    # Load num_layers if requested
    nlayers = None
    if return_nlayers:
        with h5py.File(path, "r") as f:
            if "num_layers" in f:
                nl_data = f["num_layers"][start:stop]
                # Ensure shape is (n_showers, 1)
                if nl_data.ndim == 1:
                    nl_data = nl_data[:, np.newaxis]
                nlayers = torch.from_numpy(nl_data.astype(np.float32))
            else:
                warnings.warn(
                    f"num_layers not found in {path}, returning None",
                    UserWarning,
                )

    if showers.points.shape[2] == 5:
        showers.points = showers.points[:, :, :4]
    data = ShowerDict(
        shower=torch.from_numpy(showers.points),
        energy=torch.from_numpy(showers.energies),
        direction=torch.from_numpy(showers.directions),
        pdg=torch.from_numpy(showers.pdg),
        noise=torch.from_numpy(noise) if noise is not None else None,
        sampling_frac=sampling_frac,
        nlayers=nlayers,
    )

    return data


@torch.no_grad()
def create_label_list(
    pdg: torch.Tensor,
) -> list[int]:
    unique_pdg = pdg.unique().tolist()
    unique_pdg.sort(key=lambda x: (abs(x), -x))
    return unique_pdg


@torch.no_grad()
def to_label_tensor(
    pdg: torch.Tensor | None,
    label_list: list[int] | None = None,
) -> torch.Tensor | None:
    if pdg is None:
        return None
    if label_list is None:
        label_list = create_label_list(pdg)
    if max(pdg.shape, default=1) != pdg.numel():
        raise ValueError("pdg must be a 1D tensor.")
    pdg = pdg.view(-1)
    label_tensor = torch.zeros(pdg.shape[0], dtype=torch.int64)
    for i, label in enumerate(label_list):
        label_tensor[pdg == label] = i
    return label_tensor


@torch.no_grad()
def load_and_prepare(
    path: str,
    *,
    samples_energy_trafo: Transformation = Identity(),
    samples_coordinate_trafo: Transformation = Identity(),
    cond_trafo: Transformation = Identity(),
    sampling_frac_trafo: Transformation = Identity(),
    nlayers_trafo: Transformation = Identity(),
    start: int = 0,
    stop: int | None = None,
    return_noise: bool = False,
    noise_path: str | None = None,
    return_direction: bool = False,
    return_sampling_frac: bool = False,
    return_nlayers: bool = False,
    max_num_points: int | None = None,
    num_layers: int = -1,
    do_initialise_trafos: bool = True,
    fit_stop: int | None = None,
    trafos_file: str = "",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
) -> ModelInputDict:
    data = load_data(
        path,
        start=start,
        stop=stop,
        return_noise=return_noise,
        noise_path=noise_path,
        return_sampling_frac=return_sampling_frac,
        return_nlayers=return_nlayers,
        max_num_points=max_num_points,
    )
    mask = data["shower"][:, :, [3]] > 0

    if do_initialise_trafos:
        initialise_trafos(
            data["energy"],
            data["shower"],
            mask,
            samples_energy_trafo,
            samples_coordinate_trafo,
            cond_trafo,
            sampling_frac_trafo=sampling_frac_trafo if return_sampling_frac else None,
            sampling_frac=data["sampling_frac"],
            nlayers_trafo=nlayers_trafo if return_nlayers else None,
            nlayers=data["nlayers"],
            fit_stop=fit_stop,
            trafos_file=trafos_file,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

    energy = cond_trafo(data["energy"])
    x = torch.concat(
        [
            samples_coordinate_trafo(data["shower"][:, :, :2]),
            samples_energy_trafo(data["shower"][:, :, [3]]),
        ],
        dim=-1,
    )
    x[~mask.repeat(1, 1, 3)] = 0.0
    layer = (data["shower"][:, :, [2]] + 0.1).long()
    num_points = batched_histogram(
        data=layer.squeeze(dim=-1),
        mask=mask.squeeze(dim=-1),
        num_bins=num_layers,
    )
    label = to_label_tensor(data["pdg"])

    # Build conditioning tensor
    cond_parts = [energy]
    if return_direction:
        cond_parts.append(data["direction"])
    if return_sampling_frac and data["sampling_frac"] is not None:
        cond_parts.append(sampling_frac_trafo(data["sampling_frac"]))
    if return_nlayers and data["nlayers"] is not None:
        cond_parts.append(nlayers_trafo(data["nlayers"]))
    cond = torch.concat(cond_parts, dim=-1)

    return ModelInputDict(
        x=x,
        cond=cond,
        num_points=num_points,
        layer=layer,
        mask=mask,
        label=label if label is not None else torch.zeros(0, dtype=torch.int64),
        noise=data["noise"],
    )


def get_data_loaders(
    config_dataset: dict,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    trafos_file: str = "",
) -> tuple[DataLoader, DataLoader, dict[str, Transformation]]:
    config_dataset = config_dataset.copy()
    cache_dir = config_dataset.pop("cache_dir", None)
    # Multi-seed disjoint-block support (direct in-memory path): shift the whole [0:data_len]
    # window (train + val) by data_offset, so each seed trains on a non-overlapping consecutive
    # block [data_offset : data_offset+data_len]. data_offset=0 (default) = original behavior.
    # (Not applied on the cache path: there the offset is already baked into the cache.)
    data_offset = config_dataset.pop("data_offset", 0)

    data_len = showerdata.get_file_shape(config_dataset["path"])[0]
    if "stop" in config_dataset:
        data_len = min(data_len, config_dataset["stop"])
        del config_dataset["stop"]
    fit_stop = config_dataset.pop("fit_stop", None)
    if "val_len" in config_dataset:
        val_len = config_dataset.pop("val_len")
        # if val_len > data_len // 2:
        #     warnings.warn(
        #         f"val_len {val_len} is larger than 50% of data length {data_len // 2},"
        #         f" reducing to {data_len // 2}.",
        #         UserWarning,
        #     )
        #     val_len = min(val_len, data_len // 2)
    else:
        val_len = data_len // 10
    split = data_len - val_len

    if "samples_energy_trafo" in config_dataset:
        config_dataset["samples_energy_trafo"] = compose(
            config_dataset["samples_energy_trafo"]
        )
    if "samples_coordinate_trafo" in config_dataset:
        config_dataset["samples_coordinate_trafo"] = compose(
            config_dataset["samples_coordinate_trafo"]
        )
    if "cond_trafo" in config_dataset:
        config_dataset["cond_trafo"] = compose(config_dataset["cond_trafo"])
    if "sampling_frac_trafo" in config_dataset:
        config_dataset["sampling_frac_trafo"] = compose(
            config_dataset["sampling_frac_trafo"]
        )
    if "nlayers_trafo" in config_dataset:
        config_dataset["nlayers_trafo"] = compose(config_dataset["nlayers_trafo"])

    # Memory-mapped path: load preprocessed cache from disk
    if cache_dir and os.path.isdir(os.path.join(cache_dir, "train")):
        train_start = rank * (split // world_size)
        train_stop = (rank + 1) * (split // world_size)
        data_train = MMapDictDataSet(
            os.path.join(cache_dir, "train"), start=train_start, stop=train_stop
        )
        print(f"[rank {rank}] Loaded train from mmap cache: {len(data_train)} events")
        loader_train = DataLoader(
            data_set=data_train,
            batch_size=batch_size,
            drop_last=len(data_train) > batch_size,
            shuffle=True,
        )
        if rank == 0:
            data_test = MMapDictDataSet(os.path.join(cache_dir, "val"))
            print(f"[rank {rank}] Loaded val from mmap cache: {len(data_test)} events")
            loader_test = DataLoader(
                data_set=data_test,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
            )
        else:
            loader_test = DataLoader(
                data_set=DictDataSet(
                    ModelInputDict(
                        x=torch.empty(0, 0, 0),
                        cond=torch.empty(0, 0),
                        num_points=torch.empty(0, 0, dtype=torch.int64),
                        layer=torch.empty(0, 0, dtype=torch.int64),
                        mask=torch.empty(0, 0, dtype=torch.bool),
                        label=torch.empty(0, 0, dtype=torch.int64),
                        noise=None,
                    )
                ),
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
            )
        # Load trafos for generation
        if trafos_file and os.path.isfile(trafos_file):
            parameters = torch.load(trafos_file, weights_only=True)
            for key in [
                "samples_energy_trafo",
                "samples_coordinate_trafo",
                "cond_trafo",
                "sampling_frac_trafo",
                "nlayers_trafo",
            ]:
                if key in config_dataset and key in parameters:
                    config_dataset[key].load_state_dict(parameters[key])
        trafos = {
            "samples_energy_trafo": config_dataset.get(
                "samples_energy_trafo", Identity()
            ),
            "samples_coordinate_trafo": config_dataset.get(
                "samples_coordinate_trafo", Identity()
            ),
            "cond_trafo": config_dataset.get("cond_trafo", Identity()),
            "sampling_frac_trafo": config_dataset.get(
                "sampling_frac_trafo", Identity()
            ),
            "nlayers_trafo": config_dataset.get("nlayers_trafo", Identity()),
        }
        return loader_train, loader_test, trafos

    # Original in-memory path
    start = data_offset + rank * (split // world_size)
    stop = data_offset + (rank + 1) * (split // world_size)
    data_train = DictDataSet(
        load_and_prepare(
            **config_dataset,
            start=start,
            stop=stop,
            fit_stop=fit_stop,
            trafos_file=trafos_file,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
        )
    )
    loader_train = DataLoader(
        data_set=data_train,
        batch_size=batch_size,
        drop_last=(stop - start) > batch_size,
        shuffle=True,
    )
    if rank == 0:
        data_test = DictDataSet(
            load_and_prepare(
                **config_dataset,
                start=data_offset + split,
                stop=data_offset + data_len,
                trafos_file=trafos_file,
                do_initialise_trafos=False,
            )
        )
        loader_test = DataLoader(
            data_set=data_test, batch_size=batch_size, drop_last=False, shuffle=False
        )
    else:
        loader_test = DataLoader(
            data_set=DictDataSet(
                ModelInputDict(
                    x=torch.empty(0, 0, 0),
                    cond=torch.empty(0, 0),
                    num_points=torch.empty(0, 0, dtype=torch.int64),
                    layer=torch.empty(0, 0, dtype=torch.int64),
                    mask=torch.empty(0, 0, dtype=torch.bool),
                    label=torch.empty(0, 0, dtype=torch.int64),
                    noise=None,
                )
            ),
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
        )
    trafos = {
        "samples_energy_trafo": config_dataset.get("samples_energy_trafo", Identity()),
        "samples_coordinate_trafo": config_dataset.get(
            "samples_coordinate_trafo", Identity()
        ),
        "cond_trafo": config_dataset.get("cond_trafo", Identity()),
        "sampling_frac_trafo": config_dataset.get("sampling_frac_trafo", Identity()),
        "nlayers_trafo": config_dataset.get("nlayers_trafo", Identity()),
    }
    return loader_train, loader_test, trafos
