import argparse
import pandas as pd
import json
import re
import os.path as osp
from compile import (
    compile_mlir_to_enzyme,
    jit_mlir,
    compile_c,
    compile_enzyme,
    compile_mlir,
    link_and_run,
    run_grad,
)
from jinja2 import Environment, PackageLoader, select_autoescape
import numpy as np

np.random.seed(0)
ROUND_TO = 2

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
c_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())
filepat = re.compile(r"ba\d+_n(\d+)_m(\d+)_p(\d+).txt")


def generate_results_main(args):
    import os.path as osp
    import os
    from tqdm import tqdm

    mlir_template = mlir_env.get_template("ba.mlir")
    driver_template = c_env.get_template("ba_driver.c")
    config = {
        "nCamParams": 11,
        "rot_idx": 0,
        "c_idx": 3,
        "f_idx": 6,
        "x0_idx": 7,
        "rad_idx": 9,
    }
    for ffile in tqdm(os.listdir("benchmarks/data/ba")):
        m = filepat.match(ffile)
        if not m:
            continue
        config["n"] = m.group(1)
        config["m"] = m.group(2)
        config["p"] = m.group(3)
        config["data_file"] = f"benchmarks/data/ba/{ffile}"
        config["results_file"] = f"benchmarks/results/ba/sparsemat_{ffile}"

        compile_mlir(mlir_template.render(**config).encode("utf-8"), "ba_mlir.o")
        compile_c(driver_template.render(**config).encode("utf-8"), "ba_driver.o")

        link_and_run(
            [
                "ba_mlir.o",
                # The enzyme objects are out of date but this should be fine because they aren't called.
                "enzyme_ba_reproj.o",
                "enzyme_ba_w.o",
                "ba_driver.o",
                "mlir_c_abi.o",
            ],
            "ba_driver.out",
            link_runner_utils=True,
        )


def main(args):
    data_file = osp.basename(args.data_file)
    mat = filepat.match(data_file)
    assert mat, "Malformed data file name"
    n, m, p = mat.group(1), mat.group(2), mat.group(3)
    config = {
        "data_file": f"benchmarks/data/ba/{data_file}",
        "results_file": f"benchmarks/results/ba/sparsemat_{data_file}",
        "nCamParams": 11,
        "n": n,
        "m": m,
        "p": p,
        "rot_idx": 0,
        "c_idx": 3,
        "f_idx": 6,
        "x0_idx": 7,
        "rad_idx": 9,
    }
    mlir_template = mlir_env.get_template("ba.mlir")
    enzyme_template = mlir_env.get_template("ba_enzyme_reproj.mlir")
    enzyme_w_template = mlir_env.get_template("ba_enzyme_w.mlir")
    driver_template = c_env.get_template("ba_driver.c")
    helper_template = c_env.get_template("mlir_c_abi.c")
    if args.print:
        print(mlir_template.render(**config))
        return

    compile_mlir(mlir_template.render(**config).encode("utf-8"), "ba_mlir.o")
    compile_mlir_to_enzyme(
        enzyme_template.render(**config).encode("utf-8"),
        "enzyme_ba_reproj.o",
        emit="obj",
    )
    compile_mlir_to_enzyme(
        enzyme_w_template.render(**config).encode("utf-8"), "enzyme_ba_w.o", emit="obj"
    )
    compile_c(driver_template.render(**config).encode("utf-8"), "ba_driver.o")
    compile_c(helper_template.render(**config).encode("utf-8"), "mlir_c_abi.o")
    stdout = link_and_run(
        [
            "ba_mlir.o",
            "enzyme_ba_reproj.o",
            "enzyme_ba_w.o",
            "ba_driver.o",
            "mlir_c_abi.o",
        ],
        "ba_driver.out",
        link_runner_utils=True,
    ).decode("utf-8")
    outfile = None
    try:
        lines = stdout.splitlines()
        keys = [
            "lagrad_jacobian",
            "enzyme_jacobian",
        ]
        assert len(keys) == len(
            lines
        ), f"expected # of apps to match {len(keys)} vs {len(lines)}"
        results = pd.DataFrame.from_dict(
            {key: json.loads(line) for key, line in zip(keys, lines)}
        )
        print(results.to_csv(sep="\t", index=False))

        # TODO: Meant to serialize the output results to a file.
        # if outfile:
        #     serialized_config = config.copy()
        #     # results_dict = {"config": serialized_config, "results": results.to_dict()}
        #     # with open(outfile, "w") as f:
        #     #     json.dump(results_dict, f)
        return results
    except (json.JSONDecodeError, Exception):
        print("Failed to parse output")
        print(stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_file",
        help="The name of the data file. Only the base filename will be used, as the script assumes the location in benchmarks/data/ba/...",
    )
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print the result of rendering the MLIR template, then exit.",
    )
    args = parser.parse_args()

    # main(args)
    # generate_results_main(args)
    n_runs = 3
    n_methods = 2
    datasets = [ds for ds in ["A", "B", "C"] for _ in range(n_methods)]
    methods = ["LAGrad", "Enzyme"] * 3
    run_labels = ["run1", "run2", "run3"]
    data = np.arange(3 * 2 * 3).reshape(3, 6)
    columns = pd.MultiIndex.from_tuples(zip(datasets, methods))
    df = pd.DataFrame(data=data, columns=columns, index=run_labels)
    print(df)

    # This is the code snippet to read the multindex that we want.
    # pd.read_csv('test.csv', header=[0,1], index_col=0)
