import argparse
import logging
import math
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import showerdata
import torch
import torch.nn.functional as F
import yaml
from pointcountfm.generator import Sampler, init_sampler
from torch import Tensor

from allshowers.generator import Generator


def to_labels(pdg_codes: torch.Tensor, pdgs: Iterable[int]) -> torch.Tensor:
    labels = torch.full(pdg_codes.shape, -1, dtype=torch.int64)
    for label, pdg in enumerate(pdgs):
        labels[pdg_codes == pdg] = label
    return labels


def prepare_conditions(
    energies: Tensor,
    directions: Tensor,
    labels: Tensor,
    num_particles: int,
    correct_directions: bool = False,
) -> torch.Tensor:
    cond_list = [energies.reshape(-1, 1)]
    one_hot = F.one_hot(labels, num_classes=num_particles).to(torch.float32)
    cond_list.append(one_hot)
    cond_list.append(directions)
    result = torch.concatenate(cond_list, dim=1)
    if correct_directions:
        result[:, -3:-1] *= -1.0
    return result


def generate(
    cnf_transformer: Generator,
    point_count_fm: Sampler,
    energies: Tensor,
    directions: Tensor,
    labels: Tensor,
    num_particles: int,
) -> Tensor:
    point_count_fm_input = prepare_conditions(
        energies, directions, labels, num_particles
    )
    num_points = point_count_fm(point_count_fm_input)
    # clip to non-negative and round to nearest integer
    num_points = (num_points.clamp(min=0) + 0.5).to(torch.int64)
    samples = cnf_transformer(energies, num_points, directions, labels)
    return samples


def generate_in_batches(
    cnf_transformer: Generator,
    point_count_fm: Sampler,
    energies: Tensor,
    directions: Tensor,
    pdg_codes: Tensor,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
    output_path: str | os.PathLike[str] | None = None,
    max_points: int = 6016,
    pdg_list: Iterable[int] = (
        11,
        -11,
        22,
        130,
        211,
        -211,
        321,
        -321,
        2112,
        -2112,
        2212,
        -2212,
    ),
) -> None:
    if batch_size is None:
        batch_size = energies.shape[0]
    if isinstance(device, str):
        device = torch.device(device)

    pdg_list = list(pdg_list)
    split_energies = torch.split(energies, batch_size, dim=0)
    split_directions = torch.split(directions, batch_size, dim=0)
    split_pdg_codes = torch.split(pdg_codes, batch_size, dim=0)

    if output_path is not None:
        showerdata.create_empty_file(
            path=output_path,
            shape=(energies.shape[0], max_points, 4),
        )

    times = []
    counter = 0
    for i, (energies_l, directions_l, pdg_codes_l) in enumerate(
        zip(split_energies, split_directions, split_pdg_codes)
    ):
        logging.info(f"starting batch {i + 1}")
        energies_l = energies_l.to(device)
        directions_l = directions_l.to(device)
        labels_l = to_labels(pdg_codes_l, pdg_list).to(device)
        start = time.time()
        samples_l = generate(
            cnf_transformer,
            point_count_fm,
            energies_l,
            directions_l,
            labels_l,
            num_particles=len(pdg_list),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 2:
            times.append(time.time() - start)

        if output_path is not None:
            showerdata.save_batch(
                data=showerdata.Showers(
                    points=samples_l.cpu().numpy(),
                    energies=energies_l.cpu().numpy(),
                    directions=directions_l.cpu().numpy(),
                    pdg=pdg_codes_l.cpu().numpy(),
                    shower_ids=np.arange(counter, counter + samples_l.shape[0]),
                ),
                path=output_path,
                start=counter,
            )
        counter += samples_l.shape[0]
        del samples_l
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    logging.info(
        f"mean time per batch: {mean(times):.2f} s +- {stdev(times) / math.sqrt(len(times)):.2f} s"
    )


def time_generation(
    cnf_transformer: Generator,
    point_count_fm: Sampler,
    pdg_code: int,
    batch_size: int,
    device: str | torch.device = "cpu",
    pdg_list: Iterable[int] = (
        11,
        -11,
        22,
        130,
        211,
        -211,
        321,
        -321,
        2112,
        -2112,
        2212,
        -2212,
    ),
) -> None:
    if isinstance(device, str):
        device = torch.device(device)

    pdg_list = list(pdg_list)
    energies = torch.full((batch_size, 1), 10, device=device)
    pdg_ids = torch.full((batch_size,), pdg_code)
    labels = to_labels(pdg_ids, pdg_list).to(device)
    directions = torch.tensor([[0.0, 0.0, 1.0]] * batch_size, device=device)

    # warmup
    num_warmup = max(2, 32 // batch_size)
    logging.info(f"running {num_warmup} warmup iterations")
    for _ in range(num_warmup):
        generate(
            cnf_transformer,
            point_count_fm,
            energies,
            directions,
            labels,
            num_particles=len(pdg_list),
        )
    logging.info("starting timing benchmark")

    for energy in range(10, 130, 10):
        energies = torch.full((batch_size, 1), energy, device=device)
        times = []
        for i in range(100):
            start = time.time()
            generate(
                cnf_transformer,
                point_count_fm,
                energies,
                directions,
                labels,
                num_particles=len(pdg_list),
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) / batch_size)
        logging.info(
            f"energy {energy} mean time: {mean(times)} +- {stdev(times) / math.sqrt(len(times))} s"
        )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate particle showers using AllShowers."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Path to the .h5 file where the generated showers should be saved. If not provided, "
            "the generated showers will not be saved to disk."
        ),
        default=None,
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Whether to disable compilation of the sampler (default: False).",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=512,
        help="The batch size to use for generation (default: 512).",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate (default: 1).",
    )
    parser.add_argument(
        "-p",
        "--pdg",
        type=int,
        default=22,
        help="PDG code of the incident particle (default: 22 for photons).",
    )
    parser.add_argument(
        "-e",
        "--energy",
        type=float,
        default=100.0,
        help="Energy of the incident particle in GeV (default: 100.0).",
    )
    parser.add_argument(
        "-d",
        "--direction",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        help="Incident particle direction as three floats (default: [0.0, 0.0, 1.0]).",
    )
    parser.add_argument(
        "--pdg-list",
        type=int,
        nargs="+",
        default=[11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212],
        help="How labels should be assigned based on PDG codes.",
    )
    parser.add_argument(
        "--cnf-directory",
        type=Path,
        default=Path("results/cnf-transformer"),
        help="Path to the directory containing the CNF transformer model (default: results/cnf-transformer).",
    )
    parser.add_argument(
        "--pointcountfm-directory",
        type=Path,
        default=Path("results/pointcountfm"),
        help="Path to the directory containing the PointCountFM model (default: results/pointcountfm).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=6016,
        help="Maximum number of points to generate per shower (default: 6016).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device run inference on (default: auto).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="Number of CPU threads to use for inference (default: auto).",
    )
    parser.add_argument(
        "--cond-file",
        type=Path,
        default=None,
        help=(
            "If provided, conditions are read from this file instead of using the "
            "command line arguments. The command line arguments --energy, --direction, "
            "--pdg and --num-samples are ignored if this argument is provided."
        ),
    )
    parser.add_argument(
        "-r",
        "--rescale-factor",
        default=1.033,
        type=float,
        help="energy rescale factor applied during generation (default: 1.033).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="midpoint",
        help=(
            "Solver to use for the CNF transformer. Can be either 'euler', 'heun' or "
            "'midpoint' (default: 'midpoint')."
        ),
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=16,
        help=("Number of timesteps to use for the CNF transformer (default: 16)."),
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help=(
            "If set, the instead of generating showers, a timing benchmark is run. "
            "In this case, the arguments '--energy', '--output', '--num-samples' and "
            "'--cond-file' are ignored. '--pdg' still controls which particle type "
            "is used."
        ),
    )
    return parser.parse_args(args)


def setup_logging(results_file: Path | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="{relativeCreated:9.0f}ms - {message:s}",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(results_file.with_suffix(".log")),
        ]
        if results_file is not None
        else [logging.StreamHandler(sys.stdout)],
        style="{",
    )
    logging.captureWarnings(True)


def args_to_yaml(args: argparse.Namespace) -> str:
    var_dict = vars(args)
    var_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in var_dict.items()}
    return yaml.safe_dump(var_dict, indent=2)[:-1]


def select_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    elif torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.warning("Apple Silicon GPU not supported yet, using CPU instead")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


@torch.inference_mode()
def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args)
    setup_logging(parsed_args.output)
    logging.info("Command line arguments:\n" + args_to_yaml(parsed_args))
    device = select_device(parsed_args.device)
    if parsed_args.num_threads:
        torch.set_num_threads(parsed_args.num_threads)
    logging.info(f"Using device: {device}")
    logging.info(f"Using {torch.get_num_threads()} CPU threads")
    logging.info(f"pyTorch version: {torch.__version__}")
    if device.type == "cuda":
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Device name: {torch.cuda.get_device_name(device)}")
    else:
        logging.info(f"Device name: {device}")

    logging.info("start loading models")
    generator = Generator(
        run_dir=parsed_args.cnf_directory,
        num_timesteps=parsed_args.num_timesteps,
        solver=parsed_args.solver,
        resize_factor=parsed_args.rescale_factor,
    )
    if device.type == "cuda":
        generator.flow.network.compile(fullgraph=True, dynamic=True)
    sampler = init_sampler(parsed_args.pointcountfm_directory, 50)
    sampler.compile(  # sampler.model.network.compile (for faster setup)
        disable=parsed_args.disable_compile,
    )
    logging.info("models loaded")

    if parsed_args.cond_file is not None:
        logging.info(f"Loading conditions from {parsed_args.cond_file}")
        inc_particles = showerdata.load_inc_particles(parsed_args.cond_file)
        energies = torch.from_numpy(inc_particles.energies).float()
        pdg_codes = torch.from_numpy(inc_particles.pdg).long()
        directions = torch.from_numpy(inc_particles.directions).float()
    else:
        energies = torch.full(
            (parsed_args.num_samples, 1), parsed_args.energy, dtype=torch.float32
        )
        pdg_codes = torch.full(
            (parsed_args.num_samples,), parsed_args.pdg, dtype=torch.int64
        )
        directions = torch.repeat_interleave(
            torch.tensor([parsed_args.direction]),
            repeats=parsed_args.num_samples,
            dim=0,
        )
    logging.info(f"initialized conditions for {energies.shape[0]} incident particles")

    generator = generator.to(device)
    generator.eval()
    sampler = sampler.to(device)
    sampler.eval()

    if not parsed_args.time:
        generate_in_batches(
            generator,
            sampler,
            energies,
            directions,
            pdg_codes,
            device=device,
            batch_size=parsed_args.batch_size,
            output_path=parsed_args.output,
            max_points=parsed_args.max_points,
            pdg_list=parsed_args.pdg_list,
        )
    else:
        time_generation(
            generator,
            sampler,
            parsed_args.pdg,
            batch_size=parsed_args.batch_size,
            device=device,
            pdg_list=parsed_args.pdg_list,
        )
    logging.info("all done")


if __name__ == "__main__":
    main()
