import argparse
import os
import platform
import sys
import time
import warnings
from typing import Any

import h5py
import numpy as np
import showerdata
import torch
import yaml
from showerdata.shift_showers import shift_layers
from torch import Tensor, nn

from allshowers import flow_matching as fm
from allshowers import transformer, util
from allshowers.data_sets import to_label_tensor
from allshowers.preprocessing import compose

start = time.perf_counter()


class Generator(nn.Module):
    def __init__(
        self,
        run_dir: str,
        num_timesteps: int = 200,
        compile: bool = False,
        solver: str = "heun",
        resize_factor: float = 1.0,
    ) -> None:
        super().__init__()

        run_params_file = os.path.join(run_dir, "conf.yaml")
        state_dict_file = os.path.join(run_dir, "weights/best.pt")
        if not os.path.exists(run_params_file):
            state_dict_file = os.path.join(run_dir, "weights/best-all.pt")
        trafo_file = os.path.join(run_dir, "preprocessing/trafos.pt")
        if not os.path.exists(trafo_file):
            trafo_file = os.path.join(run_dir, "preprocessing/trafos-all.pt")
        self.result_dir = run_dir
        self.num_timesteps = num_timesteps
        self.do_compile = compile
        self.resize_factor = resize_factor

        with open(run_params_file) as f:
            run_params = yaml.load(f, Loader=yaml.FullLoader)

        self.__init_model(run_params["model"], state_dict_file, solver=solver)
        self.__init_trafo(run_params["data"], trafo_file)
        self.to(torch.get_default_dtype())
        self.feature_last = run_params["data"].get("feature_last", False)
        self.num_layers = run_params["model"].get("num_layers", None)
        self.max_points = run_params["data"].get("max_num_points", 6016)

        # Check dimensionality to enable conditional features
        # Assuming Energy (1) + Angles (3) are basic, check for extras
        dim_inputs = run_params["model"]["dim_inputs"][-1]
        self.expects_sf = dim_inputs > 1
        self.expects_nlayers = dim_inputs > 2

    def __init_model(
        self, params: dict[str, Any], state_file: str, solver: str = "heun"
    ) -> None:
        flow_config = params.pop("flow_config") if "flow_config" in params else {}
        flow_config["solver"] = solver
        network = transformer.Transformer(**params)
        state_dict = torch.load(state_file, map_location="cpu", weights_only=True)
        trained_compiled = any("_orig_mod." in key for key in state_dict)
        if self.do_compile:
            network = torch.compile(network)
        network_compiled = hasattr(network, "_orig_mod")
        if trained_compiled and not network_compiled:
            for k in list(state_dict.keys()):
                if "_orig_mod." in k:
                    new_k = k.replace("_orig_mod.", "")
                    state_dict[new_k] = state_dict.pop(k)
        elif not trained_compiled and network_compiled:
            for k in list(state_dict.keys()):
                if "network." in k:
                    new_k = k.replace("network.", "network._orig_mod.")
                    state_dict[new_k] = state_dict.pop(k)
        self.flow = fm.CNF(network, **flow_config)  # type: ignore
        # Resize mismatched tensors (e.g. pretrained 45-layer model → 48-layer target)
        model_state = self.flow.state_dict()
        for key, pretrained_val in list(state_dict.items()):
            if key in model_state and pretrained_val.shape != model_state[key].shape:
                state_dict[key] = util.copy_overlapping_into_fresh(
                    pretrained_val, model_state[key]
                )
                print(
                    f"  [resize] {key}: {list(pretrained_val.shape)} → {list(model_state[key].shape)}"
                )
        self.flow.load_state_dict(state_dict, strict=False)

    def __init_trafo(self, params: dict[str, Any], trafo_file: str) -> None:
        self.samples_energy_trafo = compose(params.get("samples_energy_trafo"))
        self.samples_coordinate_trafo = compose(params.get("samples_coordinate_trafo"))
        self.cond_trafo = compose(params.get("cond_trafo"))
        self.sampling_frac_trafo = compose(params.get("sampling_frac_trafo"))
        self.nlayers_trafo = compose(params.get("nlayers_trafo"))

        state = torch.load(trafo_file, map_location="cpu", weights_only=True)
        self.samples_energy_trafo.load_state_dict(state["samples_energy_trafo"])
        self.samples_coordinate_trafo.load_state_dict(state["samples_coordinate_trafo"])
        self.cond_trafo.load_state_dict(state["cond_trafo"])
        if "sampling_frac_trafo" in state:
            self.sampling_frac_trafo.load_state_dict(state["sampling_frac_trafo"])
        if "nlayers_trafo" in state:
            self.nlayers_trafo.load_state_dict(state["nlayers_trafo"])

    def forward(
        self,
        energies: Tensor,
        num_points: Tensor,
        angles: Tensor,
        sampling_fraction: Tensor,
        nlayers: Tensor | None = None,
        label: Tensor | None = None,
        clamp_per_layer: bool = True,
    ) -> Tensor:
        # Build conditioning: Energy + Angles + [SF] + [NLayers]
        cond_parts = [self.cond_trafo(energies * self.resize_factor)]

        # Add Angles (Direction)
        cond_parts.append(angles)

        if self.expects_sf:
            cond_parts.append(self.sampling_frac_trafo(sampling_fraction))
        if self.expects_nlayers and nlayers is not None:
            cond_parts.append(self.nlayers_trafo(nlayers))

        condition = torch.concatenate(cond_parts, dim=-1)

        # Cap per-layer counts only when explicitly enabled.
        # This is useful for CLD/PCFM edge cases, but must stay off for
        # downstream generation that uses real G4 per-layer counts.
        if clamp_per_layer and self.num_layers and self.num_layers >= 40:
            num_points = num_points.clamp(max=2000)
        max_points = int(num_points.sum(dim=-1).max().item())
        layer = torch.zeros((condition.shape[0], max_points, 1), dtype=torch.int32)
        mask = torch.zeros((condition.shape[0], max_points, 1), dtype=torch.bool)
        for i in range(condition.shape[0]):
            total_points = int(torch.sum(num_points[i]).item())
            layer_i = torch.repeat_interleave(num_points[i])
            layer[i, :total_points, 0] = layer_i
            mask[i, :total_points, 0] = True
        layer = layer.to(condition.device)
        mask = mask.to(condition.device)
        raw_samples = self.flow.sample(
            shape=(condition.shape[0], max_points, 3),
            num_timesteps=self.num_timesteps,
            cond=condition,
            num_points=num_points,
            layer=layer,
            mask=mask,
            label=label,
        )
        samples = torch.zeros(
            (condition.shape[0], max_points, 4), device=raw_samples.device
        )
        samples[:, :, :2] = self.samples_coordinate_trafo.inverse(raw_samples[:, :, :2])
        samples[:, :, 2] = layer.squeeze(2)
        samples[:, :, 3] = self.samples_energy_trafo.inverse(raw_samples[:, :, 2])
        samples[~mask.repeat(1, 1, 4)] = 0
        return samples


def print_time(text):
    now = time.perf_counter()
    print(f"[{int(now - start):6d}s]: {text}")
    sys.stdout.flush()


def generate(
    generator: Generator,
    energies: Tensor,
    num_points: Tensor,
    angles: Tensor,
    sampling_fraction: Tensor,
    nlayers: Tensor | None = None,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
    labels: Tensor | None = None,
    clamp_per_layer: bool = True,
) -> Tensor:
    if batch_size is None:
        batch_size = energies.shape[0]
    split_energies = torch.split(energies, batch_size, dim=0)
    split_num_points = torch.split(num_points, batch_size, dim=0)
    split_angles = torch.split(angles, batch_size, dim=0)
    split_sampling_fraction = torch.split(sampling_fraction, batch_size, dim=0)
    if nlayers is not None:
        split_nlayers = torch.split(nlayers, batch_size, dim=0)
    else:
        split_nlayers = [None] * len(split_energies)
    if labels is not None:
        split_labels = torch.split(labels, batch_size, dim=0)
    else:
        split_labels = [None] * len(split_energies)

    generator = generator.to(device)
    generator.eval()
    samples = []
    for i, batch in enumerate(
        zip(
            split_energies,
            split_num_points,
            split_angles,
            split_sampling_fraction,
            split_nlayers,
            split_labels,
        )
    ):
        print_time(f"start batch {i:3d}")
        batch = [e.to(device) if e is not None else None for e in batch]
        samples_l = generator(*batch, clamp_per_layer=clamp_per_layer).cpu()
        samples.append(samples_l)
    # Pad to uniform sequence length before concatenating (different batches
    # may have different max_points from their local num_points maximum).
    if len(samples) > 1:
        max_seq = max(s.shape[1] for s in samples)
        samples = [
            torch.nn.functional.pad(s, (0, 0, 0, max_seq - s.shape[1]))
            if s.shape[1] < max_seq
            else s
            for s in samples
        ]
    samples = torch.cat(samples)
    print_time("generation done")
    return samples


def get_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates new samples")
    parser.add_argument(
        "run_dir",
        help="directory that contains the model's weights and where the generated samples should be saved",
    )
    parser.add_argument(
        "cond_file",
        help="file with the conditioning information (e.g. energies, number of points)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        default=1,
        type=int,
        help="number of samples to generate. default: 1",
    )
    parser.add_argument(
        "-b", "--batch-size", default=1024, type=int, help="default: 1024"
    )
    parser.add_argument("-t", "--num-threads", default=None, type=int)
    parser.add_argument("-d", "--device", default=None, help="device for computations")
    parser.add_argument(
        "--num-timesteps",
        default=200,
        type=int,
        help="number of timesteps for the ODE solver. default: 200",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        type=str,
        help="data type for the generated samples. default: float32",
    )
    parser.add_argument(
        "-r",
        "--rescale-factor",
        default=1.0,
        type=float,
        help="energy rescale factor applied during generation. default: 1.0",
    )
    parser.add_argument(
        "--solver",
        default="heun",
        type=str,
        help="ODE solver to use during generation. default: heun",
    )
    parser.add_argument(
        "--pdgs",
        default=[11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212],
        nargs="+",
        type=int,
        help="list of pdg codes for the labels. default: [11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212]",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory where generated samples will be saved. Defaults to run_dir.",
    )
    parser.add_argument(
        "--sampling-fractions",
        nargs="+",
        type=float,
        default=None,
        help="Target sampling fractions for grouping output files (e.g., 0.015 0.030 0.045)",
    )
    parser.add_argument(
        "--target-nlayers",
        nargs="+",
        type=int,
        default=None,
        help="Target nlayers values for grouping output files (e.g., 25 35 45)",
    )
    parser.add_argument(
        "--fixed-sf",
        type=float,
        default=None,
        help="Fixed sampling fraction when generating by nlayers",
    )
    parser.add_argument(
        "--fixed-nlayers",
        type=int,
        default=None,
        help="Fixed nlayers when generating by sampling fraction",
    )
    parser.add_argument(
        "--no-shift-layers",
        action="store_true",
        default=False,
        help="Skip the inverse shift_layers postprocessing (output stays in aligned frame)",
    )
    parser.add_argument(
        "--clamp-per-layer",
        action="store_true",
        default=False,
        help="Force the per-layer occupancy clamp (max 2000 pts/layer for >=40-layer "
        "detectors) ON even with --downstream. Tames physically-impossible "
        "PointCountFM extrapolations at low D (e.g. CLD D=100 scratch reaching "
        "~52k pts/layer vs Geant4's ~400) that otherwise crash/garble generation. "
        "No-op for well-behaved conditioning (counts already below the cap).",
    )
    parser.add_argument(
        "--allegro",
        action="store_true",
        default=False,
        help="Deprecated: use --downstream allegro instead",
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default=None,
        help="Downstream calorimeter tag (e.g. allegro, cld, simplebox). "
        "Implies --no-shift-layers, saves single file generated_<tag>.h5",
    )
    return parser.parse_args(args)


@torch.inference_mode()
def main(args: list[str] | None = None) -> None:
    parsed_args = get_args(args)
    parsed_args.pdgs.sort(key=lambda x: (abs(x), -x))
    # Back-compat: --allegro → --downstream allegro
    if parsed_args.allegro and not parsed_args.downstream:
        parsed_args.downstream = "allegro"
    if parsed_args.downstream:
        parsed_args.no_shift_layers = True
    clamp_per_layer = parsed_args.clamp_per_layer or not bool(parsed_args.downstream)
    print_time("start main")
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if parsed_args.dtype not in dtypes:
        raise ValueError(f"invalid dtype: {parsed_args.dtype}")
    dtype = dtypes[parsed_args.dtype]
    torch.set_default_dtype(dtype)
    torch.set_float32_matmul_precision("high")
    if parsed_args.num_threads:
        torch.set_num_threads(parsed_args.num_threads)
    print(yaml.dump(vars(parsed_args)), end="")
    if parsed_args.device:
        device = parsed_args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    torch.set_default_device(device)
    if "cuda" in device.lower():
        print("devise:", torch.cuda.get_device_name(torch.device(device)))
    elif device.lower() == "cpu":
        print("devise:", platform.processor())
    print("num threads:", torch.get_num_threads())
    sys.stdout.flush()

    generator = Generator(
        run_dir=parsed_args.run_dir,
        num_timesteps=parsed_args.num_timesteps,
        compile=True,
        solver=parsed_args.solver,
        resize_factor=parsed_args.rescale_factor,
    )
    with h5py.File(parsed_args.cond_file, "r") as f:
        # Handle both key naming conventions (incident_energies vs energies)
        energy_key = "incident_energies" if "incident_energies" in f else "energies"
        all_energies = f[energy_key][:]

        # --- Handle Angles ---
        angle_key = (
            "incident_directions" if "incident_directions" in f else "directions"
        )
        if angle_key in f:
            all_angles = f[angle_key][:]
        else:
            # Fallback if directions missing (e.g. older files)
            warnings.warn("No directions/angles found. Using placeholder (Z-dir).")
            all_angles = np.zeros((len(all_energies), 3), dtype=np.float32)
            all_angles[:, 2] = 1.0

        all_num_points = f["num_points_per_layer"][:]
        all_sf = f["sampling_fraction"][:]
        # Handle both pdg key naming conventions
        if "incident_pdg" in f:
            all_pdg = f["incident_pdg"][:]
        elif "pdg" in f:
            all_pdg = f["pdg"][:]
        else:
            all_pdg = None
        all_nlayers = f["num_layers"][:] if "num_layers" in f else None
        all_layer_z_pos = f["layer_z_pos"][:] if "layer_z_pos" in f else None

    # --- INFO: Print Angles Range from File ---
    # Convert Cartesian (vx, vy, vz) to Spherical (theta, phi) for logging
    # vz = cos(theta) -> theta = arccos(vz)
    # phi = arctan2(vy, vx)

    # Clip vz to [-1, 1] to avoid nan in arccos due to float precision
    vz_clamped = np.clip(all_angles[:, 2], -1.0, 1.0)
    thetas_deg = np.rad2deg(np.arccos(vz_clamped))
    phis_deg = np.rad2deg(np.arctan2(all_angles[:, 1], all_angles[:, 0]))

    print("\n" + "=" * 40)
    print(f"INFO: Angles read from {parsed_args.cond_file}")
    if len(thetas_deg) > 0:
        print(
            f"  Theta Range: [{np.min(thetas_deg):.2f}, {np.max(thetas_deg):.2f}] deg"
        )
        print(f"  Phi Range:   [{np.min(phis_deg):.2f}, {np.max(phis_deg):.2f}] deg")
    else:
        print("  WARNING: empty condition file chunk, skipping generation.")
        print("=" * 40 + "\n")
        sys.stdout.flush()
        sys.exit(0)
    print("=" * 40 + "\n")
    sys.stdout.flush()
    # ------------------------------------------

    # Determine generation mode
    sf_mode = parsed_args.sampling_fractions is not None
    nlayers_mode = parsed_args.target_nlayers is not None

    sf_flat_all = all_sf.flatten()
    nl_flat_all = all_nlayers.flatten() if all_nlayers is not None else None

    # Select samples based on mode
    if sf_mode:
        # SF mode: variable SF at fixed nlayers (or all nlayers if not specified)
        selected_indices = []
        n_per_group = parsed_args.num_samples
        fixed_nl = parsed_args.fixed_nlayers

        for sf_target in parsed_args.sampling_fractions:
            sf_mask = np.abs(sf_flat_all - sf_target) < 0.005
            if fixed_nl is not None and nl_flat_all is not None:
                nl_mask = nl_flat_all == fixed_nl
                combined_mask = sf_mask & nl_mask
            else:
                combined_mask = sf_mask
            group_indices = np.where(combined_mask)[0]
            if len(group_indices) > n_per_group:
                group_indices = group_indices[:n_per_group]
            selected_indices.extend(group_indices)
        selected_indices = np.array(selected_indices)
        mode_desc = f"SF mode: {len(selected_indices)} samples ({n_per_group} per SF)"
        if fixed_nl:
            mode_desc += f", fixed nlayers={fixed_nl}"

    elif nlayers_mode:
        # NLayers mode: variable nlayers at fixed SF
        selected_indices = []
        n_per_group = parsed_args.num_samples
        fixed_sf = parsed_args.fixed_sf

        if fixed_sf is None:
            raise ValueError("--fixed-sf is required when using --target-nlayers")
        if nl_flat_all is None:
            raise ValueError("Conditioning file has no num_layers data")

        for nl_target in parsed_args.target_nlayers:
            nl_mask = nl_flat_all == nl_target
            sf_mask = np.abs(sf_flat_all - fixed_sf) < 0.005
            combined_mask = sf_mask & nl_mask
            group_indices = np.where(combined_mask)[0]
            if len(group_indices) > n_per_group:
                group_indices = group_indices[:n_per_group]
            selected_indices.extend(group_indices)
        selected_indices = np.array(selected_indices)
        mode_desc = f"NLayers mode: {len(selected_indices)} samples ({n_per_group} per nlayers), fixed SF={fixed_sf}"

    else:
        # Fallback: take last n samples
        n_total = all_energies.shape[0]
        start_idx = max(0, n_total - parsed_args.num_samples)
        selected_indices = np.arange(start_idx, n_total)
        mode_desc = f"Fallback mode: {len(selected_indices)} samples"

    # Extract selected data
    energies = torch.from_numpy(all_energies[selected_indices])
    num_points = torch.from_numpy(all_num_points[selected_indices]).clamp(min=0)
    # Match num_points_per_layer width to model's num_layers
    model_nl = generator.num_layers
    if model_nl is not None and num_points.shape[1] != model_nl:
        if num_points.shape[1] > model_nl:
            num_points = num_points[:, :model_nl]
        else:
            padded = torch.zeros(num_points.shape[0], model_nl, dtype=num_points.dtype)
            padded[:, : num_points.shape[1]] = num_points
            num_points = padded
        print(f"Adjusted num_points_per_layer: {all_num_points.shape[1]} -> {model_nl}")
    sampling_fraction = torch.from_numpy(all_sf[selected_indices])
    angles = torch.from_numpy(all_angles[selected_indices])

    if all_pdg is not None:
        pdg = torch.from_numpy(all_pdg[selected_indices])
    else:
        pdg = torch.full(
            (len(selected_indices), 1), parsed_args.pdgs[0], dtype=torch.int32
        )
    if all_nlayers is not None:
        nlayers = torch.from_numpy(all_nlayers[selected_indices].astype(np.float32))
    else:
        nlayers = None
    layer_z_pos = (
        all_layer_z_pos[selected_indices] if all_layer_z_pos is not None else None
    )

    print_time(mode_desc)
    labels = to_label_tensor(
        pdg=pdg,
        label_list=parsed_args.pdgs,
    )

    energies = energies.to(dtype, copy=False)
    angles = angles.to(dtype, copy=False)  # Ensure angles are on correct device/dtype

    generator.eval()
    generator = generator.to(device)

    samples = generate(
        generator,
        energies,
        num_points,
        angles,
        sampling_fraction,
        nlayers,
        parsed_args.batch_size,
        device,
        labels,
        clamp_per_layer=clamp_per_layer,
    )
    # Clamp point energies to non-negative (column 3 is energy)
    samples[:, :, 3] = torch.clamp(samples[:, :, 3], min=0)

    output_directory = (
        parsed_args.out_dir if parsed_args.out_dir else parsed_args.run_dir
    )
    os.makedirs(output_directory, exist_ok=True)

    # Convert to numpy for saving (ensure CPU tensors)
    samples_np = samples.cpu().numpy()
    energies_np = energies.cpu().numpy()
    angles_np = angles.cpu().numpy()
    pdg_np = pdg.cpu().numpy()
    sf_flat = sampling_fraction.cpu().numpy().flatten()
    nlayers_flat = nlayers.cpu().numpy().flatten() if nlayers is not None else None

    # Apply inverse angular shift correction (postprocessing: inverse=True)
    # Restores the physical lateral displacement from the particle's incident angle.
    # Skip if --no-shift-layers is set (output stays in aligned/z-axis frame).
    if parsed_args.no_shift_layers:
        print(
            "INFO: --no-shift-layers set, skipping inverse shift_layers (aligned frame output)."
        )
    elif layer_z_pos is not None:
        for i in range(samples_np.shape[0]):
            samples_np[i] = shift_layers(
                shower=samples_np[i],
                direction=angles_np[i],
                layer_bottom_pos=layer_z_pos[i],
                calo_surface=0.0,
                inverse=True,
            )
    else:
        warnings.warn(
            "layer_z_pos not found in cond file. Skipping inverse shift_layers."
        )

    # Reorder points: valid (energy>0) first, padding (zeros) at end
    for i in range(samples_np.shape[0]):
        valid = samples_np[i, :, 3] > 1e-5
        samples_np[i] = np.concatenate([samples_np[i, valid], samples_np[i, ~valid]])

    # Downstream calorimeter mode: single output file, no grouping
    if parsed_args.downstream:
        tag = parsed_args.downstream
        file_path = os.path.join(output_directory, f"generated_{tag}.h5")
        showerdata.Showers(
            points=samples_np,
            energies=energies_np,
            directions=angles_np,
            pdg=pdg_np,
        ).save(file_path, overwrite=True)
        with h5py.File(file_path, "a") as f:
            f.create_dataset("sampling_fraction", data=sf_flat)
            if nlayers_flat is not None:
                f.create_dataset("num_layers", data=nlayers_flat)
        print_time(f"Saved {len(samples_np)} {tag.upper()} samples to {file_path}")
        # Per-file recipe sidecar: durable record of the exact recipe + output
        # name, never overwritten by a later generation into the same dir.
        with open(file_path.replace(".h5", ".recipe.yaml"), "w") as cf:
            yaml.dump(
                {
                    **vars(parsed_args),
                    "output_file": os.path.basename(file_path),
                    "n_saved": int(len(samples_np)),
                },
                cf,
            )
        with open(os.path.join(output_directory, "generation_config.yaml"), "w") as f:
            yaml.dump(vars(parsed_args), f)
        print_time("all done")
        return

    # Save samples based on generation mode
    if nlayers_mode:
        # NLayers mode: save by nlayers values
        target_nlayers = parsed_args.target_nlayers
        print_time(
            f"Saving samples split by {len(target_nlayers)} nlayers values: {target_nlayers}"
        )

        for nl_target in target_nlayers:
            indices = np.where(nlayers_flat == nl_target)[0]
            if len(indices) == 0:
                print(f"  Warning: No samples found for nlayers={nl_target}")
                continue

            nl_showers = showerdata.Showers(
                points=samples_np[indices],
                energies=energies_np[indices],
                directions=angles_np[indices],
                pdg=pdg_np[indices],
            )

            file_path = os.path.join(
                output_directory, f"generated_nlayers{nl_target}.h5"
            )
            nl_showers.save(file_path, overwrite=True)

            # Save sampling_fraction and num_layers as additional datasets
            with h5py.File(file_path, "a") as f:
                if "sampling_fraction" in f:
                    del f["sampling_fraction"]
                f.create_dataset("sampling_fraction", data=sf_flat[indices])
                if "num_layers" in f:
                    del f["num_layers"]
                f.create_dataset("num_layers", data=nlayers_flat[indices])

            print(f"  Saved {len(indices)} samples to {file_path}")
            with open(file_path.replace(".h5", ".recipe.yaml"), "w") as cf:
                yaml.dump(
                    {
                        **vars(parsed_args),
                        "output_file": os.path.basename(file_path),
                        "nlayers_target": int(nl_target),
                        "n_saved": int(len(indices)),
                    },
                    cf,
                )

    else:
        # SF mode (or fallback): save by SF values
        if parsed_args.sampling_fractions:
            target_sfs = sorted(parsed_args.sampling_fractions)
        else:
            target_sfs = sorted(np.unique(np.round(sf_flat, 3)))

        print_time(
            f"Saving samples split by {len(target_sfs)} target SF values: {target_sfs}"
        )

        for sf_target in target_sfs:
            indices = np.where(np.abs(sf_flat - sf_target) < 0.005)[0]
            if len(indices) == 0:
                print(f"  Warning: No samples found for SF={sf_target:.3f}")
                continue

            sf_showers = showerdata.Showers(
                points=samples_np[indices],
                energies=energies_np[indices],
                directions=angles_np[indices],
                pdg=pdg_np[indices],
            )

            # Use 3 decimal places for filename
            file_path = os.path.join(
                output_directory, f"generated_sf{sf_target:.3f}.h5"
            )
            sf_showers.save(file_path, overwrite=True)

            # Save num_layers as additional dataset
            if nlayers_flat is not None:
                with h5py.File(file_path, "a") as f:
                    if "num_layers" in f:
                        del f["num_layers"]
                    f.create_dataset("num_layers", data=nlayers_flat[indices])

            print(f"  Saved {len(indices)} samples to {file_path}")
            with open(file_path.replace(".h5", ".recipe.yaml"), "w") as cf:
                yaml.dump(
                    {
                        **vars(parsed_args),
                        "output_file": os.path.basename(file_path),
                        "sf_target": float(sf_target),
                        "n_saved": int(len(indices)),
                    },
                    cf,
                )

    # Save config yaml
    with open(os.path.join(output_directory, "generation_config.yaml"), "w") as f:
        yaml.dump(vars(parsed_args), f)

    print_time("all done")


if __name__ == "__main__":
    main()
