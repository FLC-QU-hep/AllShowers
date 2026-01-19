import argparse
import os

import yaml

from allshowers import util

job_script_template = """\
#!/bin/bash
#SBATCH --partition={partition:s}
#SBATCH --time=7-00:00:00
#SBATCH --nodes={num_nodes:d}
#SBATCH --job-name={name:s}
#SBATCH --output={result_path:s}/log/training-%j.out
#SBATCH --error={result_path:s}/log/training-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={mail:s}
#SBATCH --constraint={gpu_type:s}GPUx{num_gpus:d}

echo "job id: $SLURM_JOB_ID"
echo ""

srun --nodes {num_nodes:d} --ntasks-per-node 1 bash {result_path:s}/script.sh
"""

worker_script_template = """\
#!/bin/env bash

cd {repo_path:s}
source .venv/bin/activate

num_cpus=$(nproc --all)
num_gpus=$(nvidia-smi -L | wc -l)

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29400
export OMP_NUM_THREADS=$(($num_cpus / $num_gpus))

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

torchrun --nnodes={num_nodes:d} --nproc_per_node=$num_gpus --rdzv_id=100\\
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
        "-m",
        "--mail",
        type=str,
        default="",
        help="email address for SLURM notifications",
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params["result_path"] = util.setup_result_path(params["run_name"], args.param_file)

    os.mkdir(os.path.join(params["result_path"], "checkpoints"))
    os.mkdir(os.path.join(params["result_path"], "weights"))
    os.mkdir(os.path.join(params["result_path"], "plots"))
    os.mkdir(os.path.join(params["result_path"], "log"))
    os.mkdir(os.path.join(params["result_path"], "preprocessing"))
    os.mkdir(os.path.join(params["result_path"], "data"))

    conf_file = os.path.join(params["result_path"], "conf.yaml")
    run_file = os.path.join(params["result_path"], "run.sh")
    worker_file = os.path.join(params["result_path"], "script.sh")
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

    job_script = job_script_template.format(
        name=params["run_name"],
        result_path=params["result_path"],
        num_gpus=args.num_gpu,
        gpu_type=gpu_type,
        partition=args.partition,
        num_nodes=args.num_nodes,
        mail=args.mail,
    )
    if args.mail == "":
        job_script = job_script.replace("#SBATCH --mail-user=\n", "")
        job_script = job_script.replace("#SBATCH --mail-type=END,FAIL\n", "")
    with open(run_file, "w") as file:
        file.write(job_script)

    worker_script = worker_script_template.format(
        repo_path=repo_path,
        num_nodes=args.num_nodes,
        config=os.path.relpath(conf_file, repo_path),
    )

    with open(worker_file, "w") as file:
        file.write(worker_script)

    command = f"sbatch {run_file}"
    print(command)
    if args.run:
        print(os.popen(command).read())


if __name__ == "__main__":
    main()
