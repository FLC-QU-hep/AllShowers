"""
USAGE:
python mkresultdir.py --partition=maxgpu,allgpu --A100 -g 4 -n 1 --mem 1024G -r conf/pretrain/simplebox_pretrain.yaml

"""

import argparse
import os
import re
import shutil

import yaml

from allshowers import util

job_script_template = """\
#!/bin/bash
#SBATCH --partition={partition:s}
#SBATCH --time=7-00:00:00
#SBATCH --nodes={num_nodes:d}
#SBATCH --job-name={name:s}
#SBATCH --output={log_path:s}/training-%j.out
#SBATCH --error={log_path:s}/training-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={mail:s}
#SBATCH --constraint="{gpu_type:s}GPUx{num_gpus:d}"
{mem_line:s}{exclude_line:s}
echo "job id: $SLURM_JOB_ID"
echo ""

unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
srun --nodes {num_nodes:d} --ntasks-per-node 1 bash {result_path:s}/{worker:s}
"""

worker_script_template = """\
#!/bin/env bash

cd {repo_path:s}
source .venv/bin/activate

num_cpus=$(nproc --all)
num_gpus=$(nvidia-smi -L | wc -l)

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=$(($num_cpus / $num_gpus))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "node: $(uname -n)"
echo "number of CPUs: $num_cpus"
echo "number of GPUs: $num_gpus"
grep MemTotal /proc/meminfo

echo "number of threads: $OMP_NUM_THREADS"
echo "master address: $MASTER_ADDR"
echo "master port: $MASTER_PORT"

echo "config file: {config:s}"
echo "start time: $(date)"
echo ""

torchrun --nnodes={num_nodes:d} --nproc_per_node=$num_gpus --rdzv_id=$SLURM_JOB_ID\\
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT\\
    allshowers/train.py --ddp {config:s}
"""


def get_args():
    parser = argparse.ArgumentParser(
        description="create result directory and job script"
    )
    parser.add_argument("param_file", help="where to find the parameters")
    parser.add_argument(
        "-r", "--run", action="store_true", default=False, help="submit the job"
    )
    parser.add_argument(
        "--A100", action="store_true", default=False, help="use A100 only"
    )
    parser.add_argument(
        "--H100", action="store_true", default=False, help="use H100 only"
    )
    parser.add_argument(
        "--V100", action="store_true", default=False, help="use V100 only"
    )
    parser.add_argument(
        "-g", "--num_gpu", type=int, default=1, help="number of GPUs to use, default: 1"
    )
    parser.add_argument(
        "-n",
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes to use, default: 1",
    )
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        default="maxgpu",
        help='define SLURM partition, default:"maxgpu"',
    )
    parser.add_argument(
        "--mem",
        type=str,
        default=None,
        help="add RAM constraint to SLURM, e.g. --mem 1024G",
    )
    parser.add_argument(
        "-m",
        "--mail",
        type=str,
        default="",
        help="email address for SLURM notifications",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="",
        help="subdirectory under results/ (e.g. pretrain_lemurs)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="comma-separated list of nodes to exclude",
    )
    args = parser.parse_args()

    return args


def _make_job_script(args, params, log_path, worker_name, gpu_type):
    mem_line = f"#SBATCH --mem={args.mem}\n" if args.mem else ""
    exclude_line = f"#SBATCH --exclude={args.exclude}\n" if args.exclude else ""
    job_script = job_script_template.format(
        name=params["run_name"],
        result_path=params["result_path"],
        log_path=log_path,
        worker=worker_name,
        num_gpus=args.num_gpu,
        gpu_type=gpu_type,
        partition=args.partition,
        num_nodes=args.num_nodes,
        mail=args.mail,
        mem_line=mem_line,
        exclude_line=exclude_line,
    )
    if args.mail == "":
        job_script = job_script.replace("#SBATCH --mail-user=\n", "")
        job_script = job_script.replace("#SBATCH --mail-type=END,FAIL\n", "")
    return job_script


def main():
    args = get_args()
    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params["result_path"] = util.setup_result_path(
        params["run_name"], args.param_file, base_dir=args.base_dir
    )
    repo_path = os.path.dirname(os.path.abspath(__file__))

    if sum(int(x) for x in [args.A100, args.H100, args.V100]) > 1:
        raise ValueError("Only one GPU type can be selected at a time.")
    if args.H100:
        gpu_type = "H100&"
    elif args.A100:
        gpu_type = "A100&"
    elif args.V100:
        gpu_type = "V100&"
    else:
        gpu_type = "V100&" if args.num_nodes > 1 else ""
    has_finetune = "finetune" in params

    def _conf_text():
        """Read original yaml and set result_path (preserves formatting)."""
        with open(args.param_file) as f:
            text = f.read()
        rp_line = f"result_path: {params['result_path']}"
        if re.search(r"^result_path:", text, flags=re.MULTILINE):
            return re.sub(r"^result_path:.*$", rp_line, text, flags=re.MULTILINE)
        return f"{rp_line}\n{text}"

    def _write_worker(path, conf_path):
        with open(path, "w") as f:
            f.write(
                worker_script_template.format(
                    repo_path=repo_path,
                    num_nodes=args.num_nodes,
                    config=os.path.relpath(conf_path, repo_path),
                )
            )

    rp = params["result_path"]

    if has_finetune:
        # ── finetune + from_scratch layout ───────────────────────────────────
        for sub in ("finetune/log", "from_scratch/log"):
            os.makedirs(os.path.join(rp, sub))

        # util.setup_result_path already wrote conf.yaml — rename it
        shutil.move(
            os.path.join(rp, "conf.yaml"), os.path.join(rp, "conf_finetune.yaml")
        )
        conf_finetune = os.path.join(rp, "conf_finetune.yaml")
        conf_scratch = os.path.join(rp, "conf_scratch.yaml")

        # Symlink conf.yaml inside subdirs so generator.py finds it
        os.symlink("../conf_finetune.yaml", os.path.join(rp, "finetune", "conf.yaml"))
        os.symlink(
            "../conf_scratch.yaml", os.path.join(rp, "from_scratch", "conf.yaml")
        )

        conf_text = _conf_text()
        scratch_text = re.sub(
            r"^finetune:(\n  .*)+\n", "", conf_text, flags=re.MULTILINE
        )
        original_lr = params["train"].get("learning_rate", 1e-4)
        scratch_text = re.sub(
            r"(learning_rate:\s*)[\de.+-]+",
            rf"\g<1>{original_lr * 10:.1e}",
            scratch_text,
        )
        with open(conf_scratch, "w") as f:
            f.write(scratch_text)

        run_finetune = os.path.join(rp, "run_finetune.sh")
        run_scratch = os.path.join(rp, "run_scratch.sh")
        with open(run_finetune, "w") as f:
            f.write(
                _make_job_script(
                    args,
                    params,
                    os.path.join(rp, "finetune/log"),
                    "script_finetune.sh",
                    gpu_type,
                )
            )
        with open(run_scratch, "w") as f:
            f.write(
                _make_job_script(
                    args,
                    params,
                    os.path.join(rp, "from_scratch/log"),
                    "script_scratch.sh",
                    gpu_type,
                )
            )

        _write_worker(os.path.join(rp, "script_finetune.sh"), conf_finetune)
        _write_worker(os.path.join(rp, "script_scratch.sh"), conf_scratch)

        print(f"finetune:     sbatch {run_finetune}")
        print(f"from_scratch: sbatch {run_scratch}")
        if args.run:
            print(os.popen(f"sbatch {run_finetune}").read())

    else:
        # ── Original pretrain layout ─────────────────────────────────────────
        for sub in ("checkpoints", "weights", "plots", "log", "preprocessing", "data"):
            os.mkdir(os.path.join(rp, sub))

        conf_file = os.path.join(rp, "conf.yaml")
        run_file = os.path.join(rp, "run.sh")
        with open(conf_file, "w") as f:
            f.write(_conf_text())
        with open(run_file, "w") as f:
            f.write(
                _make_job_script(
                    args,
                    params,
                    os.path.join(rp, "log"),
                    "script.sh",
                    gpu_type,
                )
            )
        _write_worker(os.path.join(rp, "script.sh"), conf_file)

        print(f"sbatch {run_file}")
        if args.run:
            print(os.popen(f"sbatch {run_file}").read())


if __name__ == "__main__":
    main()
