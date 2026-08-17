import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import showerdata
from matplotlib.lines import Line2D

mpl.rcParams["figure.figsize"] = (13, 8)
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, amssymb}"
mpl.rcParams["font.size"] = 16
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.sans-serif"] = "Times New Roman"
mpl.rcParams["legend.frameon"] = False


@dataclass
class ParticlePlotConfig:
    particles: list[str]
    pdg_list: list[int]
    colors: list[str | tuple[float, float, float]]


NAME_TO_PDG = {
    "e-": 11,
    "e+": -11,
    "gamma": 22,
    "pi+": 211,
    "pi-": -211,
    "k+": 321,
    "k-": -321,
    "k0l": 130,
    "p": 2212,
    "p_bar": -2212,
    "n": 2112,
    "n_bar": -2112,
}
PDG_TO_NAME = {v: k for k, v in NAME_TO_PDG.items()}


def get_unique_pdg(data: dict[str, np.ndarray]) -> list[int]:
    unique_pdg = np.unique(data["incident_pdg"])
    unique_pdg = sorted(unique_pdg.tolist(), key=lambda x: (abs(x), -x))

    return unique_pdg


def name_to_tex(name: str) -> str:
    name = name.replace("+", r"^{+}")
    name = name.replace("-", r"^{-}")
    name = name.replace("0", r"^{0}")
    name = name.replace("L", r"_{L}")
    name = name.replace("l", r"_{L}")
    if "pi" in name:
        name = name.replace("pi", r"\pi")
    elif "mu" in name:
        name = name.replace("mu", r"\mu")
    elif "gamma" in name:
        name = name.replace("gamma", r"\gamma")
    elif "e" in name:
        name = name.replace("e", r"\text{e}")
    elif "p" in name:
        name = name.replace("p", r"\text{p}")
    elif "n" in name:
        name = name.replace("n", r"\text{n}")
    elif "K" in name:
        name = name.replace("K", r"\text{K}")
    elif "k" in name:
        name = name.replace("k", r"\text{K}")
    if name.endswith("_bar"):
        name = name.removesuffix("_bar")
        name = r"\overline{" + name + "}"
    return r"\(" + name + r"\)"


def filter_and_histogram(
    showers: showerdata.Showers,
    pdg: int,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.where(showers.pdg == pdg)[0][:10_000]
    hit_energies = 1e3 * showers.points[:, :, 3][idx]
    hit_energies = hit_energies[hit_energies > 0.0]
    mean = np.histogram(hit_energies, bins=bins)[0]
    mean = np.pad(mean, (1, 0), mode="edge")
    uncertainty = np.sqrt(mean)
    return mean, uncertainty


def add_ratio_plot(
    axis,
    bins: np.ndarray,
    geant4_mean: np.ndarray,
    geant4_unc: np.ndarray,
    panshower_mean: np.ndarray,
    panshower_unc: np.ndarray,
    name: str,
    color: str | tuple[float, float, float],
    vline: float | None = None,
) -> None:
    axis.hlines(1.0, bins[0], bins[-1], colors="darkgray")
    mask = geant4_mean > 2e-1
    axis.step(
        bins[mask],
        panshower_mean[mask] / geant4_mean[mask],
        color=color,
    )
    axis.fill_between(
        bins[mask],
        (
            1
            - np.sqrt(
                geant4_unc[mask] ** 2 / geant4_mean[mask] ** 2
                + panshower_unc[mask] ** 2 / panshower_mean[mask] ** 2
            )
        )
        * (panshower_mean[mask] / geant4_mean[mask]),
        (
            1
            + np.sqrt(
                geant4_unc[mask] ** 2 / geant4_mean[mask] ** 2
                + panshower_unc[mask] ** 2 / panshower_mean[mask] ** 2
            )
        )
        * (panshower_mean[mask] / geant4_mean[mask]),
        color=color,
        alpha=0.3,
        step="pre",
    )

    axis.grid(axis="y")
    axis.set_ylabel("Ratio " + name_to_tex(name))
    axis.set_ylim(0.8, 1.2)
    axis.set_yticks([0.9, 1.0, 1.1])

    if vline is not None:
        axis.vlines(vline, 0.8, 1.2, color="black", linestyle="dotted")


def add_profile(
    axes,
    observable_name: str,
    geant4_data: dict,
    panshower_data: dict,
    xlabel: str,
    config: ParticlePlotConfig,
    bin_factor: float = 1.0,
    xlim: tuple[float, float] = (2e-1, 4e2),
    vline: float | None = None,
) -> None:
    for i, (name, pdg, color) in enumerate(
        zip(config.particles, config.pdg_list, config.colors)
    ):
        geant4_mean = 1e3 * geant4_data[observable_name][
            geant4_data["incident_pdg"] == pdg
        ][:10_000].mean(axis=0)
        panshower_mean = 1e3 * panshower_data[observable_name][
            panshower_data["incident_pdg"] == pdg
        ][:10_000].mean(axis=0)
        geant4_std = 1e3 * geant4_data[observable_name][
            geant4_data["incident_pdg"] == pdg
        ][:10_000].std(axis=0)
        panshower_std = 1e3 * panshower_data[observable_name][
            panshower_data["incident_pdg"] == pdg
        ][:10_000].std(axis=0)
        geant4_mean = np.pad(geant4_mean, (1, 0), mode="edge")
        geant4_std = np.pad(geant4_std, (1, 0), mode="edge")
        panshower_mean = np.pad(panshower_mean, (1, 0), mode="edge")
        panshower_std = np.pad(panshower_std, (1, 0), mode="edge")
        geant4_unc = geant4_std / np.sqrt(10_000)
        panshower_unc = panshower_std / np.sqrt(10_000)

        axes[0].step(
            np.arange(geant4_mean.shape[0]) * bin_factor,
            geant4_mean,
            label=name_to_tex(name),
            color=color,
        )
        axes[0].step(
            np.arange(panshower_mean.shape[0]) * bin_factor,
            panshower_mean,
            color=color,
            linestyle="dashed",
        )

        add_ratio_plot(
            axes[i + 1],
            np.arange(geant4_mean.shape[0]) * bin_factor,
            geant4_mean,
            geant4_unc,
            panshower_mean,
            panshower_unc,
            name,
            color,
            vline=vline,
        )

    if vline is not None:
        axes[0].vlines(vline, xlim[0], xlim[1], color="black", linestyle="dotted")

    axes[0].set_ylim(xlim[0], xlim[1])
    axes[0].set_xlim(0, (geant4_mean.shape[0] - 1) * bin_factor)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean Energy [MeV]")
    axes[4].set_xlabel(xlabel)


def add_cell_energy(
    axes,
    geant4_showers: showerdata.Showers,
    panshower_showers: showerdata.Showers,
    config: ParticlePlotConfig,
) -> None:
    bins = np.logspace(-1, 3, 101)
    for i, (name, pdg, color) in enumerate(
        zip(config.particles, config.pdg_list, config.colors)
    ):
        geant4_mean, geant4_unc = filter_and_histogram(geant4_showers, pdg, bins)
        panshower_mean, panshower_unc = filter_and_histogram(
            panshower_showers, pdg, bins
        )

        axes[0].step(
            bins,
            geant4_mean,
            label=name_to_tex(name),
            color=color,
        )
        axes[0].step(
            bins,
            panshower_mean,
            color=color,
            linestyle="dashed",
        )

        add_ratio_plot(
            axes[i + 1],
            bins,
            geant4_mean,
            geant4_unc,
            panshower_mean,
            panshower_unc,
            name,
            color,
        )

    axes[0].set_xlim(bins[0], bins[-1])
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_ylabel(r"\# Cells")
    axes[4].set_xlabel("Cell Energy [MeV]")


def sum_bins(data: np.ndarray, factor: int) -> np.ndarray:
    new_size = (data.shape[0], (data.shape[1] + factor - 1) // factor, factor)
    data = np.pad(
        data,
        ((0, 0), (0, new_size[1] * factor - data.shape[1])),
        mode="constant",
        constant_values=0.0,
    )
    data = data.reshape(new_size).sum(axis=2)
    return data


def add_legend(axis, config: ParticlePlotConfig) -> None:
    handles = [
        Line2D([0], [0], color=color, lw=2, label=name_to_tex(name))
        for name, color in zip(config.particles, config.colors)
    ]
    handles.extend(
        [
            Line2D([0], [0], color="black", linestyle="solid", label="Geant4"),
            Line2D([0], [0], color="black", linestyle="dashed", label="AllShowers"),
        ]
    )
    axis.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
    )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot comparison of multiple particle types simulated with Geant4 and AllShowers."
    )
    parser.add_argument(
        "geant4_path",
        type=Path,
        help="Path to Geant4 shower data HDF5 file.",
    )
    parser.add_argument(
        "allshowers_path",
        type=Path,
        help="Path to AllShowers shower data HDF5 file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path.cwd() / "plots",
        help="Path to save the output plots.",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args)

    geant4_data = showerdata.observables.read_observables_from_file(
        parsed_args.geant4_path
    )
    panshower_data = showerdata.observables.read_observables_from_file(
        parsed_args.allshowers_path
    )
    geant4_showers = showerdata.load(parsed_args.geant4_path)
    panshower_showers = showerdata.load(parsed_args.allshowers_path)

    geant4_data["energy_per_radial_bin"] = sum_bins(
        geant4_data["energy_per_radial_bin"], factor=2
    )
    panshower_data["energy_per_radial_bin"] = sum_bins(
        panshower_data["energy_per_radial_bin"], factor=2
    )

    particles = ["e-", "gamma", "pi-", "n"]
    # particles = ["e+", "pi+", "p", "p_bar"]
    # particles = ["n_bar", "k+", "k-", "k0L"]
    pdg_list = [NAME_TO_PDG[name.lower()] for name in particles]
    config = ParticlePlotConfig(
        particles=particles,
        pdg_list=pdg_list,
        colors=["blue", "red", "brown", "green"],
    )

    fig, axes = plt.subplots(5, 3, height_ratios=[3, 1, 1, 1, 1], sharex="col")
    fig.subplots_adjust(
        hspace=0,
        wspace=0.3,
    )

    add_cell_energy(
        axes[:, 0],
        geant4_showers,
        panshower_showers,
        config,
    )
    add_profile(
        axes[:, 1],
        "energy_per_layer",
        geant4_data,
        panshower_data,
        "Layer",
        config,
        vline=30.0,
    )
    add_profile(
        axes[:, 2],
        "energy_per_radial_bin",
        geant4_data,
        panshower_data,
        "Radius [mm]",
        config,
        bin_factor=4.0,
        xlim=(5e-1, 2e4),
    )
    add_legend(axes[0, 2], config)

    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    fig.align_ylabels(axes[:, 2])
    fig.align_xlabels(axes[4, :])
    fig.savefig(
        parsed_args.output_path / "multi_particle_types.pdf", bbox_inches="tight"
    )
    fig.savefig(
        parsed_args.output_path / "multi_particle_types.png", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
