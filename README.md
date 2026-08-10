# Transferable Fast Calorimeter Shower Generation via Multi-Geometry Pre-training

[![arXiv](https://img.shields.io/badge/arXiv-2608.18233-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.18233)
[![Python Version](https://img.shields.io/badge/Python_3.13-306998?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch_2.8-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/FLC-QU-hep/AllShowers?tab=MIT-1-ov-file)
[![Build Status](https://img.shields.io/github/actions/workflow/status/FLC-QU-hep/AllShowers/pre_commit.yaml?label=pre-commit&logo=github)](https://github.com/FLC-QU-hep/AllShowers/actions/workflows/pre_commit.yaml)
[![Tests](https://img.shields.io/github/actions/workflow/status/FLC-QU-hep/AllShowers/test.yaml?label=tests&logo=github)](https://github.com/FLC-QU-hep/AllShowers/actions/workflows/test.yaml)

Code release (`multi-geometry` branch) for the paper
**Transferable Fast Calorimeter Shower Generation via Multi-Geometry Pre-training**
(T. Buss, H. Day-Hall, F. Gaede, G. Kasieczka, K. Krüger, P. McKeown, L. Valente).
The model is a single conditional flow matching model for calorimeter
shower point clouds. A geometry-aware conditioning on the sampling fraction
$f_s$ and the number of layers $N_\mathrm{layers}$ lets one backbone be
pre-trained across many geometries and then transferred to unseen detectors
with little target data. The pre-training pools are the synthetic SimpleBox
family and the realistic LEMURS detectors. The held-out transfer target is
FCCee-ALLEGRO. The code builds on the
[AllShowers](https://arxiv.org/abs/2601.11716) backbone (this repository's
`main` branch).

The multi-geometry datasets are published in a single research-data record
([DOI 10.25592/uhhfdm.19103](https://doi.org/10.25592/uhhfdm.19103)) and the
pre-trained weights on Hugging Face
([FLC-QU-hep/AllShowers-multi-geometry](https://huggingface.co/FLC-QU-hep/AllShowers-multi-geometry)).
The configurations of the paper's pre-trainings and fine-tunings are in `conf/`.


## Steps to install and run the code
### 1. Clone this repository:
```bash
git clone https://github.com/FLC-QU-hep/AllShowers.git
cd AllShowers
```
### 2. Install dependencies
To install the required dependencies, chose **one** of the following options:

#### Using uv (option 1)
```bash
uv sync --group=dev
source .venv/bin/activate
```

#### Using pip + venv (option 2)
```bash
python3.13 -m venv --prompt AllShowers .venv
source .venv/bin/activate
pip install -e .
pip install --group dev
```
If you do not have python 3.13 installed, you can try with a different version, but be aware that the code has only been tested with python 3.13.

#### Using conda (option 3)
```bash
conda env create -f environment.yaml
conda activate AllShowers
pip install -e .
```
This will install all large packages from conda-forge and some smaller, pure Python packages that are not available from via conda, from PyPI. The last line should only install AllShowers itself.

#### pre-commit hooks
After installing the dependencies, you can install the pre commit hooks with:
```bash
pre-commit install
```
This will automatically format your code when you create a git commit.

### 3. Download datasets
The multi-geometry datasets (the SimpleBox pool, the LEMURS detectors and the
FCCee-ALLEGRO target) are published in a single research-data record:
[DOI 10.25592/uhhfdm.19103](https://doi.org/10.25592/uhhfdm.19103). Download the
files you need and place them under `data/` following the paths expected by the
config files in `conf/`. The LEMURS pre-training pool is published as four
per-detector files (`lemurs_*_1M.h5`). `conf/pretrain/lemurs_pretrain.yaml` expects them
concatenated into a single `data/pretrain_lemurs/lemurs_4M.h5`, with the
OT-matched latent points then generated locally via `allshowers/OT_match.py`. The dataset of the original single-detector paper remains
available on [Zenodo](https://zenodo.org/records/18020348) and is documented on
the `main` branch.

You can generate optimal transport matched latent points with the `allshowers/OT_match.py` script. On MacOS you might need to deactivate file locking for HDF5 first. For the full dataset, the matching is computationally expensive. It will run parallel on multiple cores.
```bash
# only needed on some filesystems (e.g. Apple File System)
export HDF5_USE_FILE_LOCKING=FALSE

# run OT matching (pick the config of the pool you train on)
python allshowers/OT_match.py conf/pretrain/lemurs_pretrain.yaml
```
The latent points will be stored in the same h5 file. Preprocessing and data file path will be read from the config file. If preprocessing transformations or data path change, you need to re-run the OT matching.

If you do not want to use OT matched latent points, you need to set `return_noise: False` in the data section of the config file. Some of the unit tests will fail if you do not compute OT matched latent points first.

### 4. Run tests (optional)
Now you can run the unit tests to check that everything is working correctly:
```bash
python -m unittest discover -s test -p "*_test.py" -v
```

### 5. Run training
The `conf/` folder ships the configurations used in the multi-geometry paper,
organised by study:
- `conf/pretrain/`: the three pre-trainings (`simplebox_pretrain.yaml`,
  `lemurs_pretrain.yaml`, `simplebox_mini_pretrain.yaml`)
- `conf/allegro_finetuning/`: the transfer to FCCee-ALLEGRO, one folder per
  arm of the paper's comparison (`from_simplebox/`, `from_lemurs/`,
  `from_simplebox_mini/`, `from_scratch/`), each at the four dataset sizes
  (`D100.yaml` to `D100k.yaml`)
- `conf/cld_finetuning/`, `conf/odd_finetuning/`, `conf/par04_scipb_finetuning/`,
  `conf/par04_siw_finetuning/`: the SimpleBox-pretrained fine-tuning on the
  four Si/Sci detectors (`D*.yaml`) with the from-scratch baselines alongside
  (`scratch_D*.yaml`), at the same four sizes
- `conf/simplebox_finetuning/`: the held-out SimpleBox-like target, same
  pattern.

Every file is the as-run recipe of the corresponding paper run. The fine-tuning
configs point to a pre-trained checkpoint in their `finetune` section. Set it
to your own pre-training run or to the released weights. The `scratch_*` and
`from_scratch/` configs are the corresponding baselines: same recipe without
the `finetune` block and with `train.learning_rate` at `1.e-3`. When running several sizes of the same
target, give each run its own `result_path` (the configs of one detector share
it by default). For example, to pre-train on the LEMURS pool:
```bash
python allshowers/train.py conf/pretrain/lemurs_pretrain.yaml
```
The code loads the entire dataset into memory to speed up training. If you do not have enough memory, you might need to reduce the size of the dataset or modify the data loading code to load data in batches from disk.
For testing purposes, you can run a very short training with the flag `--fast-dev-run`. On a two core CPU, this still takes round about half an hour.
```bash
python allshowers/train.py --fast-dev-run conf/transformer.yaml
```

For distributing training on multiple GPUs and/or multiple nodes with SLURM, you might find the `mkresultdir.py` script useful.

### 6. Generate new samples
After training, you can generate samples with:
```bash
python allshowers/generator.py -n <num_samples> --num-timesteps 16 --solver midpoint results/<run_name> <condition>
```
here,
- `<num_samples>` is the number of samples you want to generate
- `<run_name>` is the name of the training run you want to use for generation, run `ls results/` to see all available runs
- `<condition>` is a path to a `showerdata` file containing the incident particles, the per-layer point counts (`num_points_per_layer`) and the sampling fraction (`sampling_fraction`). [PointCountFM](https://github.com/FLC-QU-hep/PointCountFM) produces such files. Alternatively both quantities can be taken from a Geant4 test file (see `.github/workflows/test.yaml` for a minimal example of preparing one with `showerdata add-observables`).

The generated samples will be stored in the run directory as `generated_<condition-name>.h5`.

### 7. Evaluate generated samples
You can calculate observables from the generated samples with:
```bash
showerdata add-observables <showerdata_file>
```

To read out the observables from python, you can use the `showerdata` package:
```python
import showerdata

path = "<showerdata_file>"
showers = showerdata.load(path)
observables = showerdata.observables.read_observables_from_file(path)
for key in observables:
    print(f"{key}: {observables[key].dtype}, {observables[key].shape}")
print()
print(f"Points shape: {showers.points.shape}")
print(f"Energies shape: {showers.energies.shape}")
print(f"Directions shape: {showers.directions.shape}")

```

### 8. Timing
The code for compiling and timing everything including PointCountFM is not yet in the repository.


### 9. Try out you own configurations
You can try out your own configurations by modifying the config files in the `conf/` folder.
```bash
mkdir conf-test
cp conf/transformer.yaml conf-test/my_transformer.yaml
```
The `conf-test` folder is in the `.gitignore` by default, so you can safely modify and create new config files there without affecting the repository.

The configuration options are described int the configuration section.

## Configuration
The configuration files are written in YAML format. You can find an example in `conf/transformer.yaml`.

### Global
- `run_name`: Name of the training run, will be part of the result folder name and set as the job-name in SLURM when using `mkresultdir.py`

### Data

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Path to the HDF5 data file |
| `samples_energy_trafo` | list | List of transformations applied to point energies |
| `samples_coordinate_trafo` | list | List of transformations applied to point coordinates |
| `cond_trafo` | list | List of transformations applied to incident energies |
| `return_noise` | boolean | Whether to use OT matched latent space points (has to be stored in the data file) |
| `noise_path` | string | Optional separate HDF5 file holding the OT matched latent points |
| `return_direction` | boolean | Whether to condition on the incident-particle direction |
| `return_sampling_frac` | boolean | Whether to condition on the sampling fraction $f_s$ (multi-geometry) |
| `return_nlayers` | boolean | Whether to condition on the number of layers $N_\mathrm{layers}$ (multi-geometry) |
| `sampling_frac_trafo` | list | Transformations applied to the sampling-fraction conditioner |
| `nlayers_trafo` | list | Transformations applied to the number-of-layers conditioner |
| `val_len` | integer | Validation set size |
| `stop` | integer | Optional stop index when only a subset of the data should be used for training |

**Transformation types** can include:
- `Affine`: Linear transformation with scale and shift parameters. example: `[Affine, {scale: 2, shift: 0.0}]`
- `Log`: Logarithmic transformation with alpha (for numerical stability) and base (default `math.e`). example: `[Log, {alpha: 0.0, base: 10}]`
- `LogIt`: Logit transformation with alpha for numerical stability. example: `[LogIt, {alpha: 0.001}]`
- `StandardScaler`: Standardization with specified shape. The Shape parameter is a list of integers with as many entries as the number of dimensions of the data to be transformed. Each entry is either 1 (take mean and standard deviation along this dimension) or the size of the dimension (do not take mean and standard deviation along this dimension). The mean and standard deviation values will be computed from the first 100k samples in the training data. example: `[StandardScaler, {shape: [1, 1, 2]}]`

### Model

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_layers` | integer | Number of calorimeter layers |
| `dim_inputs` | list | Dimensions of input features [point features, 2x Fourier frequencies, kinematic features] |
| `dim_embedding` | integer | Dimension of the embedding space |
| `num_head` | integer | Number of attention heads in multi-head attention |
| `num_blocks` | integer | Number of transformer blocks |
| `dim_feedforward` | integer | Dimension of feedforward network |
| `num_points_cond` | integer | Hidden layer size for num points conditioning |
| `activation` | string | Activation function (e.g., GELU, ReLU) |
| `num_layer_cond` | integer | Number of calorimeter layers points can attend to in addition to their own layer |
| `num_particles` | integer | Number of incident particle types |
| `flow_config.frequencies` | integer | Number of frequencies for Fourier feature encoding |

### Training

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `num_epochs` | integer | Number of training epochs | |
| `learning_rate` | float | Initial learning rate | |
| `batch_size` | integer | Batch size for training | |
| `optimizer` | string | Optimizer type (SGD, Adam, AdamW or Ranger) | AdamW |
| `scheduler` | string | Learning rate scheduler type (Step, Exponential, OneCycle, Cosine, CosineWarmup) | None |
| `weight_decay` | float | Weight decay for optimizer | 0.0 |
| `grad_clip` | float | Euclidean norm for gradient clipping | None |
| `grad_accum` | integer | Number of gradient accumulation steps | 1 |
| `momentum` | float | Momentum for SGD optimizer. Ignored for other optimizers. | 0.0 |


---
For questions/comments about the code contact: thorsten.buss@uni-hamburg.de<br/>
For questions about this `multi-geometry` branch contact: lorenzo.valente@uni-hamburg.de

The `multi-geometry` branch was written for the paper:

**Transferable Fast Calorimeter Shower Generation via Multi-Geometry Pre-training**<br/>
[https://arxiv.org/abs/2608.18233](https://arxiv.org/abs/2608.18233)<br/>
*Thorsten Buss, Henry Day-Hall, Frank Gaede, Gregor Kasieczka, Katja Krüger, Peter McKeown and Lorenzo Valente*

The AllShowers backbone was written for the paper:

**AllShowers: One model for all calorimeter showers**<br/>
[https://arxiv.org/abs/2601.11716](https://arxiv.org/abs/2601.11716)<br/>
*Thorsten Buss, Henry Day-Hall, Frank Gaede, Gregor Kasieczka and Katja Krüger*
