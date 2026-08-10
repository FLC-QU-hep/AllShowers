"""Chunked preprocessing for memory-mapped DDP training.

Processes the dataset in chunks (~200k events at a time) so it runs on any
node with ~100 GB RAM.  Saves train/val splits as .npy memmap files that can
be opened by multiple DDP ranks with near-zero RAM overhead.

Usage:
    python -u allshowers/preprocess_cache.py conf/pretrain/lemurs_pretrain.yaml
"""

import argparse
import os
import sys
import time

import numpy as np
import showerdata
import torch
import yaml

from allshowers.data_sets import initialise_trafos, load_and_prepare, load_data
from allshowers.preprocessing import Identity, compose


def _fit_transforms(
    path,
    split,
    fit_stop,
    chunk_size,
    data_cfg,
    kwargs,
    trafos_file,
    t0,
    max_fit_events=200_000,
    data_offset=0,
):
    """Fit transforms on small slices evenly spaced across the dataset.

    Loads max_fit_events total, spread over n_slices positions so all
    detector regions are represented without loading the full dataset.
    """
    return_sf = data_cfg.get("return_sampling_frac", False)
    return_nl = data_cfg.get("return_nlayers", False)

    n_slices = max(1, fit_stop // chunk_size)
    slice_size = min(chunk_size, max_fit_events // n_slices)
    offsets = [int(i * split / n_slices) for i in range(n_slices)]

    parts = {"shower": [], "energy": [], "mask": [], "sampling_frac": [], "nlayers": []}
    for off in offsets:
        end = min(off + slice_size, split)
        raw = load_data(
            path,
            start=data_offset + off,
            stop=data_offset + end,
            return_sampling_frac=return_sf,
            return_nlayers=return_nl,
        )
        parts["shower"].append(raw["shower"])
        parts["energy"].append(raw["energy"])
        parts["mask"].append(raw["shower"][:, :, [3]] > 0)
        if raw.get("sampling_frac") is not None:
            parts["sampling_frac"].append(raw["sampling_frac"])
        if raw.get("nlayers") is not None:
            parts["nlayers"].append(raw["nlayers"])
        del raw
        print(f"  fit slice {off:>8d} – {end:>8d}  [{time.time() - t0:.0f}s]")
        sys.stdout.flush()

    cat = {k: torch.cat(v) if v else None for k, v in parts.items()}
    del parts

    n_fit = len(cat["energy"])
    initialise_trafos(
        cat["energy"],
        cat["shower"],
        cat["mask"],
        kwargs["samples_energy_trafo"],
        kwargs["samples_coordinate_trafo"],
        kwargs["cond_trafo"],
        sampling_frac_trafo=kwargs.get("sampling_frac_trafo"),
        sampling_frac=cat["sampling_frac"],
        nlayers_trafo=kwargs.get("nlayers_trafo"),
        nlayers=cat["nlayers"],
        fit_stop=n_fit,
        trafos_file=trafos_file,
    )
    del cat
    print(
        f"Transforms fitted on {n_slices} slices × {slice_size} events [{time.time() - t0:.0f}s]"
    )
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"].copy()
    cache_dir = data_cfg.pop("cache_dir", None)
    if not cache_dir:
        print("ERROR: config data.cache_dir not set.", file=sys.stderr)
        sys.exit(1)

    path = data_cfg["path"]
    file_shape = showerdata.get_file_shape(path)
    data_len = min(file_shape[0], data_cfg.get("stop", file_shape[0]))
    val_len = data_cfg.get("val_len", data_len // 10)
    split = data_len - val_len
    # Multi-seed disjoint-block support: shift the WHOLE [0:data_len] window (train [0:split] +
    # val [split:data_len]) by data_offset, so each seed caches a non-overlapping block
    # [data_offset : data_offset+data_len] of the file. Memmap WRITE indices stay local ([0:split]);
    # only the file READ positions shift. data_offset=0 (default) = original behavior.
    data_offset = data_cfg.get("data_offset", 0)
    if data_offset + data_len > file_shape[0]:
        print(
            f"ERROR: data_offset={data_offset} + data_len={data_len} exceeds file ({file_shape[0]}).",
            file=sys.stderr,
        )
        sys.exit(1)
    if data_offset:
        print(
            f"[multi-seed] caching disjoint block [{data_offset} : {data_offset + data_len}] "
            f"(train [{data_offset}:{data_offset + split}], val [{data_offset + split}:{data_offset + data_len}])"
        )
    num_layers = config["model"]["num_layers"]

    result_path = config.get("result_path", "results")
    trafos_dir = os.path.join(result_path, "from_scratch", "preprocessing")
    os.makedirs(trafos_dir, exist_ok=True)
    trafos_file = os.path.join(trafos_dir, "trafos.pt")

    # Build kwargs for load_and_prepare (reusable across chunks)
    kwargs = {"path": path, "num_layers": num_layers}
    for key in [
        "samples_energy_trafo",
        "samples_coordinate_trafo",
        "cond_trafo",
        "sampling_frac_trafo",
        "nlayers_trafo",
    ]:
        kwargs[key] = compose(data_cfg[key]) if key in data_cfg else Identity()
    for key in [
        "return_noise",
        "noise_path",
        "return_direction",
        "return_sampling_frac",
        "return_nlayers",
    ]:
        if key in data_cfg:
            kwargs[key] = data_cfg[key]

    chunk_size = args.chunk_size
    fit_stop = data_cfg.get("fit_stop", chunk_size)
    t0 = time.time()

    # ── Step 1a: fit transforms on representative sample from across dataset ──
    _fit_transforms(
        path,
        split,
        fit_stop,
        chunk_size,
        data_cfg,
        kwargs,
        trafos_file,
        t0,
        data_offset=data_offset,
    )

    # ── Step 1b: first chunk → discover shapes, start memmaps ──
    first_end = min(chunk_size, split)
    print(f"Loading first chunk (0 – {first_end}) …")
    sys.stdout.flush()
    first = load_and_prepare(
        **kwargs,
        start=data_offset,
        stop=data_offset + first_end,
        do_initialise_trafos=False,
        trafos_file=trafos_file,
    )

    # ── Step 2: allocate train memmaps ──
    train_dir = os.path.join(cache_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    mmaps: dict[str, np.memmap] = {}
    for key, val in first.items():
        if val is not None:
            arr = val.numpy()
            shape = (split,) + arr.shape[1:]
            fpath = os.path.join(train_dir, f"{key}.npy")
            mm = np.lib.format.open_memmap(
                fpath, mode="w+", dtype=arr.dtype, shape=shape
            )
            mm[:first_end] = arr
            mm.flush()
            mmaps[key] = mm
            print(
                f"  {key}: {shape}  {arr.dtype}  ({np.prod(shape) * arr.itemsize / 1e9:.1f} GB)"
            )
    del first
    print(f"Train memmaps allocated, first chunk written [{time.time() - t0:.0f}s]")
    sys.stdout.flush()

    # ── Step 3: remaining train chunks ──
    for cs in range(chunk_size, split, chunk_size):
        ce = min(cs + chunk_size, split)
        chunk = load_and_prepare(
            **kwargs,
            start=data_offset + cs,
            stop=data_offset + ce,
            do_initialise_trafos=False,
            trafos_file=trafos_file,
        )
        for key, val in chunk.items():
            if val is not None:
                mmaps[key][cs:ce] = val.numpy()
        del chunk
        for mm in mmaps.values():
            mm.flush()
        print(f"  train chunk {cs:>8d} – {ce:>8d}  [{time.time() - t0:.0f}s]")
        sys.stdout.flush()

    del mmaps

    # ── Step 4: validation set (small, regular npy) ──
    print(f"Processing validation set ({split} – {data_len}) …")
    sys.stdout.flush()
    val_dir = os.path.join(cache_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    val = load_and_prepare(
        **kwargs,
        start=data_offset + split,
        stop=data_offset + data_len,
        do_initialise_trafos=False,
        trafos_file=trafos_file,
    )
    for key, v in val.items():
        if v is not None:
            np.save(os.path.join(val_dir, f"{key}.npy"), v.numpy())
    del val

    # ── Report ──
    total_bytes = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(cache_dir)
        for f in fs
    )
    print(
        f"Done [{time.time() - t0:.0f}s].  Cache: {total_bytes / 1e9:.0f} GB  →  {cache_dir}"
    )


if __name__ == "__main__":
    with torch.no_grad():
        main()
