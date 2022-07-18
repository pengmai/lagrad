import argparse
import pandas as pd
import json
import re
import os
import pathlib
from compile import (
    compile_mlir_to_enzyme,
    compile_c,
    compile_enzyme,
    compile_mlir,
    link_and_run,
)
from jinja2 import Environment, PackageLoader, select_autoescape

mlir_env = Environment(loader=PackageLoader("mlir"), autoescape=select_autoescape())
c_env = Environment(loader=PackageLoader("C"), autoescape=select_autoescape())


def main(args):
    driver_template = c_env.get_template("hand_driver.c")
    helper_template = c_env.get_template("mlir_c_abi.c")
    # enzyme_c_template = c_env.get_template("enzyme_hand_rowmaj.c")
    enzyme_c_packed_template = c_env.get_template("enzyme_hand_packed.c")
    enzyme_c_template = c_env.get_template("enzyme_hand.c")
    # enzyme_mlir_template = mlir_env.get_template("hand_enzyme.mlir")
    enzyme_mlir_template = mlir_env.get_template("hand_complicated_enzyme.mlir")
    lagrad_template = mlir_env.get_template("hand.mlir")
    # lagrad_template = mlir_env.get_template("hand_differentiated.mlir")
    # lagrad_template = mlir_env.get_template("hand_inlined.mlir")
    # data_file = args.data_file
    # data_file = "benchmarks/data/hand/hand_complicated.txt"
    enzyme_home = (
        pathlib.Path.home()
        / "Research"
        / "Enzyme"
        / "enzyme"
        / "benchmarks"
        / "hand"
        / "data"
        / "complicated_big"
    )
    # data_file = "benchmarks/data/hand/complicated_small/hand1_t26_c100.txt"
    data_file = enzyme_home / "hand1_t26_c100.txt"
    model_dir = enzyme_home / "model"
    with open(data_file) as f:
        npts, ntheta = [int(x) for x in f.readline().split()]
        assert ntheta == 26, "Unsupported value for ntheta"
    with open(model_dir / "vertices.txt") as f:
        nverts = sum(1 for _ in f)
    nbones = 22
    config = {
        "nbones": nbones,
        "ntheta": 26,
        "nverts": nverts,
        "npts": npts,
        "ntriangles": 1084,
        "primal_shape": f"{npts}x3",
        # "model_dir": "benchmarks/data/hand/complicated_small/model",
        "model_dir": model_dir,
        "data_file": data_file,
        "complicated": True,
    }

    if args.print:
        print(enzyme_mlir_template.render(**config))
        # print(lagrad_template.render(**config))
        return

    compile_mlir_to_enzyme(
        enzyme_mlir_template.render(**config).encode("utf-8"),
        "hand_enzyme_mlir.o",
        emit="obj",
    )
    compile_c(driver_template.render(**config).encode("utf-8"), "hand_driver.o")
    compile_c(helper_template.render(**config).encode("utf-8"), "mlir_c_abi.o")
    # compile_enzyme(
    #     enzyme_c_packed_template.render(**config).encode("utf-8"),
    #     "hand_enzyme_packed.o",
    # )
    # compile_enzyme(enzyme_c_template.render(**config).encode("utf-8"), "hand_enzyme.o")
    compile_mlir(lagrad_template.render(**config).encode("utf-8"), "hand_lagrad.o")
    stdout = link_and_run(
        [
            "hand_driver.o",
            "mlir_c_abi.o",
            "memusage.o",
            "hand_enzyme.o",
            "hand_lagrad.o",
            "hand_enzyme_mlir.o",
            "hand_enzyme_packed.o",
        ],
        "hand_driver.out",
        link_runner_utils=True,
    ).decode("utf-8")

    try:
        lines = stdout.splitlines()
        keys = [
            "LAGrad",
            "Enzyme/MLIR",
            # "Enzyme/C",
        ]
        assert len(keys) == len(
            lines
        ), f"expected # of apps to match {len(keys)} vs {len(lines)}"
        results = pd.DataFrame.from_dict(
            {key: json.loads(line) for key, line in zip(keys, lines)}
        )
        print(results)
        print(results[1:].mean())
        print(
            f"Speedup: {results[1:].mean()['Enzyme/MLIR'] / results[1:].mean()['LAGrad']}"
        )
        return results
    except (json.JSONDecodeError, Exception):
        print("Failed to parse output")
        print(stdout)


def get_datasets(data_dir):
    pat = re.compile(r"hand(\d+).+\.txt")

    def sorter(str):
        m = pat.match(str)
        return int(m.group(1))

    datasets = [ffile for ffile in os.listdir(data_dir) if pat.match(ffile)]
    datasets.sort(key=sorter)
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print the result of rendering the MLIR template, then exit.",
    )
    parser.add_argument("--data_file")
    args = parser.parse_args()
    data_dir = "benchmarks/data/hand/simple_small"
    datasets = get_datasets(data_dir)

    # with tqdm(datasets) as t:
    #     for dataset in t:
    #         t.write(f"Benchmarking {dataset}")
    #         args.data_file = f"{data_dir}/{dataset}"
    #         t.write(main(args))

    main(args)
