import argparse
import multiprocessing
import os
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import ot
import showerdata
import torch
import yaml

from allshowers import preprocessing

start = time.time()
batch_type = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.int64]
]


def print_time(*args, **kwargs) -> None:
    elapsed = time.time() - start
    print(f"[{elapsed: 5.2f}s]", *args, **kwargs)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match noise to points using OT and save it to the file."
            "The mapping is done for each shower and each layer separately."
        )
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output HDF5 file for OT noise. If not set, writes back to the source file.",
    )
    return parser.parse_args(args)


class PreProcessor:
    def __init__(self, config_file: str) -> None:
        with open(config_file) as file:
            config = yaml.safe_load(file)
        self.samples_energy_trafo = preprocessing.compose(
            transformation=config["data"]["samples_energy_trafo"],
        )
        self.samples_coordinate_trafo = preprocessing.compose(
            transformation=config["data"]["samples_coordinate_trafo"],
        )
        self.file_path, showers, self.data_shape = self.__get_data(config)
        showers = torch.from_numpy(showers)
        mask = showers[:, :, 3] > 0.0
        self.samples_coordinate_trafo.to(showers.dtype)
        self.samples_energy_trafo.to(showers.dtype)
        self.samples_coordinate_trafo.fit(
            x=showers[:, :, :2],
            mask=mask[:, :, None].repeat(1, 1, 2),
        )
        self.samples_energy_trafo.fit(
            x=showers[:, :, 3],
            mask=mask,
        )
        layer = (showers[:, :, 2] + 0.5).to(torch.int64)
        self.num_layers = int(torch.max(layer).item() + 1)

    def __get_data(
        self, config: dict[str, Any]
    ) -> tuple[str, npt.NDArray[np.float32], tuple[int, ...]]:
        stop = config["data"].get("stop", 100_000)
        fit_stop = min(config["data"].get("fit_stop", stop), stop)
        showers = showerdata.load(
            path=config["data"]["path"],
            stop=fit_stop,
        )
        points = showers.points[:, :, :4]
        # data_shape reflects the full dataset for noise allocation
        if fit_stop < stop:
            with h5py.File(config["data"]["path"], "r") as f:
                file_shape = f["shape"][:]  # [n_samples, max_points, features]
            data_shape = (stop, int(file_shape[1]), points.shape[2])
        else:
            data_shape = (points.shape[0], points.shape[1], points.shape[2])
        return config["data"]["path"], points, data_shape

    def __call__(
        self,
        x: npt.NDArray[np.float32],
    ) -> batch_type:
        x_tensor = torch.from_numpy(x)
        mask = x_tensor[:, 3] > 0.0
        x_tensor[:, :2] = self.samples_coordinate_trafo(
            x_tensor[:, :2].permute(0, 2, 1)
        ).permute(0, 2, 1)
        x_tensor[:, 3] = self.samples_energy_trafo(x_tensor[:, 3])
        layer = (x_tensor[:, 2] + 0.5).to(torch.int64)
        x_tensor = x_tensor[:, [0, 1, 3]]
        return x_tensor.numpy(), mask.numpy(), layer.numpy()


class DataLoader(Iterable[npt.NDArray[np.float32]]):
    def __init__(self, data_file: str, batch_size: int) -> None:
        self.file_name = data_file
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[npt.NDArray[np.float32]]:
        with showerdata.ShowerDataFile(self.file_name, "r") as file:
            for start in range(0, len(file), self.batch_size):
                end = min(start + self.batch_size, len(file))
                samples = file[start:end].points
                yield samples.transpose(0, 2, 1)


class NoiseMatcher:
    def __init__(self, pre_processor: PreProcessor) -> None:
        self.__num_layers = pre_processor.num_layers
        self.pre_processor = pre_processor

    def __call__(self, samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        points, mask, layer = self.pre_processor(samples)
        noise = np.random.randn(points.shape[0], 3, points.shape[2])

        for i in range(self.__num_layers):
            mask_local = np.expand_dims(np.logical_and(mask, layer == i), 1)
            for j in range(len(points)):
                points_j = (
                    points[j].T[mask_local[j].repeat(3).reshape(-1, 3)].reshape(-1, 3)
                )
                noise_j = (
                    noise[j].T[mask_local[j].repeat(3).reshape(-1, 3)].reshape(-1, 3)
                )
                if len(points_j) > 1:
                    N = len(points_j)
                    assert len(noise_j) == N
                    M = np.sqrt(
                        np.sum(
                            (points_j[:, None, :] - noise_j[None, :, :]) ** 2, axis=-1
                        )
                    )
                    wa = np.ones(N) / N
                    wb = np.ones(N) / N
                    T = ot.emd(wa, wb, M)
                    noise_j = N * (T @ noise_j)
                    noise[j].T[mask_local[j].repeat(3).reshape(-1, 3)] = (
                        noise_j.flatten()
                    )
        noise[(~mask[:, None, :]).repeat(3, axis=1)] = 0.0
        return noise.astype(np.float32, copy=False)


def process_file(
    data_file,
    data_shape: tuple[int, ...],
    pre_processor: PreProcessor,
    batch_size: int = 1024,
    output_file: str | None = None,
) -> None:
    output_file = output_file or data_file
    num_batches = -(-data_shape[0] // batch_size)
    print_time("batch size:", batch_size)
    print_time("number of batches:", num_batches)
    sys.stdout.flush()

    noise_matcher = NoiseMatcher(pre_processor)
    noise_shape = (data_shape[0], 3, data_shape[1])
    noise_tmp = output_file + ".tmp.dat"
    noise = np.memmap(noise_tmp, dtype=np.float32, mode="w+", shape=noise_shape)
    print_time(f"NoiseMatcher initialized. (noise memmap shape={noise.shape})")
    sys.stdout.flush()

    num_processes = min(n - 1 if (n := os.process_cpu_count()) else 1, 32)
    print_time(f"Using {num_processes} worker processes.")
    sys.stdout.flush()
    with multiprocessing.Pool(num_processes) as pool:
        for i, batch in enumerate(
            pool.imap(
                noise_matcher,
                DataLoader(data_file, batch_size),
            )
        ):
            noise[i * batch_size : i * batch_size + len(batch)] = batch
            if (i + 1) % 100 == 0:
                noise.flush()
                print_time(f"  batch {i + 1}/{num_batches}")
                sys.stdout.flush()
    noise.flush()
    print_time("All batches processed.")
    sys.stdout.flush()

    noise_t = noise.transpose(0, 2, 1)
    with h5py.File(data_file, "r") as f:
        num_points = f["num_points"][: data_shape[0]]
    showerdata.save_target(noise_t, output_file, num_points=num_points, overwrite=True)

    del noise, noise_t
    os.unlink(noise_tmp)
    print_time(f"Noise saved successfully to {output_file}.")
    sys.stdout.flush()


@torch.inference_mode()
def main(args: list[str] | None = None):
    torch.set_num_threads(1)

    parsed_args = parse_args(args)
    print_time("Parsing arguments:", parsed_args)
    sys.stdout.flush()

    pre_processor = PreProcessor(parsed_args.file)
    print_time("PreProcessor initialized.")
    sys.stdout.flush()

    print_time("Processing file")
    sys.stdout.flush()
    output_file = parsed_args.output or pre_processor.file_path
    process_file(
        data_file=pre_processor.file_path,
        data_shape=pre_processor.data_shape,
        pre_processor=pre_processor,
        batch_size=1024,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
