import datetime
import os
import shutil

from torch import Tensor

__all__ = ["copy_overlapping_into_fresh", "setup_result_path"]


def copy_overlapping_into_fresh(pretrained_val: Tensor, target_val: Tensor) -> Tensor:
    """Warm-start a shape-mismatched tensor.

    Return a fresh clone of ``target_val`` (keeping its own initialisation) with
    the region that overlaps ``pretrained_val`` copied over. Used when a
    pretrained checkpoint tensor and the target model tensor differ in shape
    (e.g. a 45-layer embedding warm-starting a 48-layer model): the overlapping
    slice is transferred and the extra rows/cols keep their fresh init.
    """
    new_val = target_val.clone()
    slices = tuple(
        slice(0, min(p, t)) for p, t in zip(pretrained_val.shape, target_val.shape)
    )
    new_val[slices] = pretrained_val[slices]
    return new_val


def setup_result_path(
    run_name: str, conf_file: str, fast_dev_run: bool = False, base_dir: str = ""
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)

    now = datetime.datetime.now()
    while True:
        full_run_name = now.strftime("%Y%m%d_%H%M%S") + "_" + run_name
        result_path = os.path.join(repo_dir, "results", base_dir, full_run_name)
        if not os.path.exists(result_path):
            if not fast_dev_run:
                os.makedirs(result_path)
            else:
                result_path = os.path.join(repo_dir, "results/test")
                if os.path.exists(result_path):
                    shutil.rmtree(result_path)
                os.makedirs(result_path)
            break
        else:
            now += datetime.timedelta(seconds=1)

    with open(conf_file) as f:
        content_list = f.readlines()

    content_list = [line for line in content_list if not line.startswith("result_path")]
    content_list.insert(1, f"result_path: {result_path}\n")
    content = "".join(content_list)

    with open(os.path.join(result_path, "conf.yaml"), "w") as f:
        f.write(content)

    return result_path
