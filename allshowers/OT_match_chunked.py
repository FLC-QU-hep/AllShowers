"""
Chunked OT matching for large datasets (e.g., LEMURS 4M showers).

Three modes:
  1) fit-scalers  — load full data, fit scalers, save to pickle
  2) process-chunk — load pre-fitted scalers, OT-match one chunk, save .npy
  3) merge         — combine all chunk .npy files into final showerdata HDF5

Usage:
  # Step 1: fit scalers (single job, needs RAM for full dataset)
  python -m allshowers.OT_match_chunked fit-scalers conf/pretrain/lemurs_pretrain.yaml \\
      --scaler-file data/pretrain_lemurs/ot_scalers.pkl

  # Step 2: process chunks (SLURM array, each job is lightweight)
  python -m allshowers.OT_match_chunked process-chunk conf/pretrain/lemurs_pretrain.yaml \\
      --scaler-file data/pretrain_lemurs/ot_scalers.pkl \\
      --chunk-id 0 --chunk-size 2000 \\
      --output-dir data/pretrain_lemurs/ot_chunks

  # Step 3: merge all chunks
  python -m allshowers.OT_match_chunked merge conf/pretrain/lemurs_pretrain.yaml \\
      --output-dir data/pretrain_lemurs/ot_chunks \\
      --output data/pretrain_lemurs/lemurs_ot_noise.h5
"""

import argparse
import multiprocessing
import os
import pickle
import sys
import time

import h5py
import numpy as np
import ot
import showerdata
import torch
import yaml

from allshowers import preprocessing

start_time = time.time()


def print_time(*args, **kwargs):
    elapsed = time.time() - start_time
    print(f"[{elapsed:7.1f}s]", *args, **kwargs)
    sys.stdout.flush()


def load_config(config_file):
    with open(config_file) as f:
        return yaml.safe_load(f)


def fit_and_save_scalers(config_file, scaler_file, fit_stop=100_000):
    """Load a subset for scaler fitting, read full dataset dims from file."""
    config = load_config(config_file)
    data_path = config["data"]["path"]
    total_stop = config["data"].get("stop", 100_000)

    # Read full dataset dimensions without loading all data
    with h5py.File(data_path, "r") as f:
        n_samples = min(total_stop, f["showers"].shape[0])
        shape_ds = f["shape"][:]  # [n_showers, max_points, n_features]
        n_points = int(shape_ds[1])
    print_time(f"Full dataset: {n_samples} samples, {n_points} max points")

    # Load subset for scaler fitting
    fit_n = min(fit_stop, n_samples)
    print_time(f"Loading {fit_n} showers for scaler fitting...")
    showers = showerdata.load(path=data_path, stop=fit_n)
    points = showers.points[:, :, :4]
    print_time(f"Loaded: shape={points.shape}")

    points_t = torch.from_numpy(points)
    mask = points_t[:, :, 3] > 0.0

    energy_trafo = preprocessing.compose(
        transformation=config["data"]["samples_energy_trafo"]
    )
    coord_trafo = preprocessing.compose(
        transformation=config["data"]["samples_coordinate_trafo"]
    )
    energy_trafo.to(points_t.dtype)
    coord_trafo.to(points_t.dtype)
    coord_trafo.fit(x=points_t[:, :, :2], mask=mask[:, :, None].repeat(1, 1, 2))
    energy_trafo.fit(x=points_t[:, :, 3], mask=mask)

    layer = (points_t[:, :, 2] + 0.5).to(torch.int64)
    num_layers = int(torch.max(layer).item() + 1)

    scaler_data = {
        "energy_trafo": energy_trafo,
        "coord_trafo": coord_trafo,
        "num_layers": num_layers,
        "data_path": data_path,
        "n_samples": n_samples,
        "n_points": n_points,
    }

    os.makedirs(os.path.dirname(scaler_file), exist_ok=True)
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler_data, f)
    print_time(
        f"Scalers saved to {scaler_file} (num_layers={num_layers}, n_samples={n_samples}, n_points={n_points})"
    )


def load_scalers(scaler_file):
    with open(scaler_file, "rb") as f:
        return pickle.load(f)


class ChunkedPreProcessor:
    """PreProcessor that uses pre-fitted scalers."""

    def __init__(self, scaler_data):
        self.energy_trafo = scaler_data["energy_trafo"]
        self.coord_trafo = scaler_data["coord_trafo"]
        self.num_layers = scaler_data["num_layers"]

    def __call__(self, x):
        x_tensor = torch.from_numpy(x)
        mask = x_tensor[:, 3] > 0.0
        x_tensor[:, :2] = self.coord_trafo(x_tensor[:, :2].permute(0, 2, 1)).permute(
            0, 2, 1
        )
        x_tensor[:, 3] = self.energy_trafo(x_tensor[:, 3])
        layer = (x_tensor[:, 2] + 0.5).to(torch.int64)
        x_tensor = x_tensor[:, [0, 1, 3]]
        return x_tensor.numpy(), mask.numpy(), layer.numpy()


class NoiseMatcher:
    def __init__(self, pre_processor):
        self.__num_layers = pre_processor.num_layers
        self.pre_processor = pre_processor

    def __call__(self, samples):
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


def process_chunk(config_file, scaler_file, chunk_id, chunk_size, output_dir):
    """Process one chunk of showers and save matched noise."""
    torch.set_num_threads(1)

    scaler_data = load_scalers(scaler_file)
    pre_processor = ChunkedPreProcessor(scaler_data)
    noise_matcher = NoiseMatcher(pre_processor)

    chunk_start = chunk_id * chunk_size
    chunk_end = min(chunk_start + chunk_size, scaler_data["n_samples"])
    actual_size = chunk_end - chunk_start

    if actual_size <= 0:
        print_time(
            f"Chunk {chunk_id}: empty (start={chunk_start} >= n_samples={scaler_data['n_samples']})"
        )
        return

    print_time(f"Chunk {chunk_id}: loading showers [{chunk_start}:{chunk_end}]")

    with showerdata.ShowerDataFile(scaler_data["data_path"], "r") as f:
        samples = f[chunk_start:chunk_end].points  # (actual_size, n_points, 4+)
    samples = samples[:, :, :4].transpose(0, 2, 1)  # (actual_size, 4, n_points)

    print_time(
        f"Chunk {chunk_id}: loaded {actual_size} showers, starting OT matching..."
    )

    # Process in sub-batches for multiprocessing
    batch_size = 256
    num_batches = -(-actual_size // batch_size)
    noise = np.empty((actual_size, 3, samples.shape[2]), dtype=np.float32)

    num_processes = min(n - 1 if (n := os.process_cpu_count()) else 1, 16)
    print_time(
        f"Chunk {chunk_id}: using {num_processes} workers, {num_batches} batches"
    )

    def batch_iter():
        for i in range(0, actual_size, batch_size):
            yield samples[i : i + batch_size]

    with multiprocessing.Pool(num_processes) as pool:
        for i, batch in enumerate(pool.imap(noise_matcher, batch_iter())):
            noise[i * batch_size : i * batch_size + len(batch)] = batch
            if (i + 1) % 5 == 0:
                print_time(f"  Chunk {chunk_id}: batch {i + 1}/{num_batches}")

    noise = noise.transpose(0, 2, 1)  # (actual_size, n_points, 3)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"ot_chunk_{chunk_id:05d}.npy")
    np.save(out_path, noise)
    print_time(f"Chunk {chunk_id}: saved {out_path} ({noise.shape})")


def merge_chunks(config_file, output_dir, output_file):
    """Merge chunk .npy files into final HDF5, streaming to avoid OOM."""
    config = load_config(config_file)

    chunk_files = sorted(
        f
        for f in os.listdir(output_dir)
        if f.startswith("ot_chunk_") and f.endswith(".npy")
    )
    print_time(f"Found {len(chunk_files)} chunk files in {output_dir}")

    # Read first chunk to get n_points dimension
    first = np.load(os.path.join(output_dir, chunk_files[0]))
    n_points = first.shape[1]
    del first

    # Count total samples
    n_total = 0
    for cf in chunk_files:
        arr = np.load(os.path.join(output_dir, cf), mmap_mode="r")
        n_total += arr.shape[0]
    print_time(f"Total samples: {n_total}, n_points: {n_points}")

    stop = config["data"].get("stop", 100_000)
    with h5py.File(config["data"]["path"], "r") as f:
        num_points = f["num_points"][:stop]

    assert len(num_points) == n_total, (
        f"num_points ({len(num_points)}) != n_total ({n_total})"
    )

    # Write directly to H5 chunk by chunk (no full concatenation in RAM)
    # Use showerdata vlen format: convert dense noise to vlen per shower
    vlen_dt = h5py.special_dtype(vlen=np.float32)
    with h5py.File(output_file, "w") as out:
        out.create_dataset("target/point_clouds", shape=(n_total,), dtype=vlen_dt)
        out.create_dataset("target/num_points", data=num_points)

        idx = 0
        for i, cf in enumerate(chunk_files):
            arr = np.load(os.path.join(output_dir, cf))  # (chunk, n_points, 3)
            for j in range(arr.shape[0]):
                npts = int(num_points[idx])
                out["target/point_clouds"][idx] = arr[j, :npts, :].flatten()
                idx += 1
            if (i + 1) % 50 == 0 or i == len(chunk_files) - 1:
                print_time(
                    f"  Written {idx}/{n_total} samples ({i + 1}/{len(chunk_files)} chunks)"
                )
            del arr

    print_time(f"Saved final OT noise to {output_file} ({n_total} samples)")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_fit = sub.add_parser("fit-scalers")
    p_fit.add_argument("config", type=str)
    p_fit.add_argument("--scaler-file", required=True, type=str)

    p_chunk = sub.add_parser("process-chunk")
    p_chunk.add_argument("config", type=str)
    p_chunk.add_argument("--scaler-file", required=True, type=str)
    p_chunk.add_argument("--chunk-id", required=True, type=int)
    p_chunk.add_argument("--chunk-size", required=True, type=int)
    p_chunk.add_argument("--output-dir", required=True, type=str)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("config", type=str)
    p_merge.add_argument("--output-dir", required=True, type=str)
    p_merge.add_argument("--output", required=True, type=str)

    args = parser.parse_args()

    if args.mode == "fit-scalers":
        fit_and_save_scalers(args.config, args.scaler_file)
    elif args.mode == "process-chunk":
        process_chunk(
            args.config,
            args.scaler_file,
            args.chunk_id,
            args.chunk_size,
            args.output_dir,
        )
    elif args.mode == "merge":
        merge_chunks(args.config, args.output_dir, args.output)


if __name__ == "__main__":
    main()
