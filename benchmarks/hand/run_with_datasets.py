import pathlib

import ronin.utils.messages as rum


def myannounce(message, prefix="rōnin", color="green"):
    pass


rum.announce = myannounce

from ronin.ninja import NinjaFile
from ronin.contexts import new_context
from ronin_phases.memory_monitor import run_with_memory
from subprocess import check_call, check_output, DEVNULL
import json
import pandas as pd
import numpy as np
import re
from build import get_project
from tqdm import tqdm

data_dir = (
    pathlib.Path.home()
    / "Research"
    / "Enzyme"
    / "enzyme"
    / "benchmarks"
    / "hand"
    / "data"
    / "simple_big"
)

model_dir = data_dir / "model"

RESULTS_DIR = pathlib.Path.home() / "Research" / "lagrad" / "detailed_results"

runtime_results_file = RESULTS_DIR / "hand_sparse_runtimes.tsv"
mem_results_file = RESULTS_DIR / "hand_memusage.tsv"
filepat = re.compile(r"hand(?P<order>\d+)_t\d+_c(?P<npts>\d+).txt")
datasets = [p for p in data_dir.glob("hand*")]


def sorter(p: pathlib.Path):
    m = filepat.match(p.name)
    return int(m.group("order"))


datasets.sort(key=sorter)


def make_runtime_dataframe(datasets, out_file):
    n_cols = 6
    # methods = ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]
    methods = ["LAGrad"]
    run_labels = [f"run{i+1}" for i in range(n_cols)]
    idx = pd.MultiIndex.from_product(
        ([ds.name for ds in datasets], methods, ["Simple", "Complicated"])
    )
    n_rows = len(idx)
    df = pd.DataFrame(
        data=np.zeros((n_rows, n_cols), dtype=np.int32), columns=run_labels, index=idx
    )
    df.to_csv(out_file, sep="\t")
    print(df)


# make_runtime_dataframe(datasets, runtime_results_file)

complicated = "complicated" in data_dir.name
print(f"Complicated: {complicated}")
with tqdm(datasets) as t:
    runtime_df = pd.read_csv(runtime_results_file, sep="\t", index_col=[0, 1, 2])
    with new_context() as ctx:
        for dataset in t:
            t.write(f"Running {dataset.name}")
            project = get_project({"npts": 0})
            binary = (
                f"{project.output_path}/bin/{project.phases['Link Executable'].output}"
            )
            stdout = check_output(
                [binary, model_dir, dataset, str(int(complicated))]
            ).decode("utf-8")
            t.write(stdout)
            lines = stdout.splitlines()
            for line in lines:
                key, results = line.split(":")
                runtime_df.loc[
                    # dataset.name, key, "Complicated" if complicated else "Simple"
                    dataset.name,
                    key,
                    "Big",
                ] = json.loads(results)
            runtime_df.to_csv(runtime_results_file, sep="\t")
