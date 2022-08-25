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
from build import get_packed_project

from tqdm import tqdm

data_dir = (
    pathlib.Path.home()
    / "Research"
    / "Enzyme"
    / "enzyme"
    / "benchmarks"
    / "gmm"
    / "data"
    / "10k"
)

runtime_results_file = (
    pathlib.Path.home()
    / "Research"
    / "lagrad"
    / "detailed_results"
    / "gmm_packed_runtimes.tsv"
)

mem_results_file = (
    pathlib.Path.home()
    / "Research"
    / "lagrad"
    / "detailed_results"
    / "gmm_memusage.tsv"
)

filepat = re.compile(r"gmm_d(?P<d>\d+)_K(?P<k>\d+).txt")
# Subview bug prevents running d = 2
datasets = [
    p for p in data_dir.glob("gmm_*") if int(filepat.match(p.name).group("d")) != 2
]


def sorter(p):
    m = filepat.match(p.name)
    # sizes.k + sizes.d * sizes.k * 2 + sizes.k * sizes.d ** 2
    k, d = int(m.group("k")), int(m.group("d"))
    return k + k * d + k * d * (d + 1) / 2


def make_dataframe(datasets, out_file):
    n_runs = 6
    methods = ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]
    copied_datasets = [ds.name for ds in datasets for _ in range(len(methods))]
    n_cols = len(copied_datasets)
    copied_methods = methods * len(datasets)
    run_labels = [f"run{i+1}" for i in range(n_runs)]
    data = np.zeros(n_cols * n_runs, dtype=np.int32).reshape(n_cols, n_runs)
    idx = pd.MultiIndex.from_tuples(zip(copied_datasets, copied_methods))
    df = pd.DataFrame(data=data, columns=run_labels, index=idx)
    df.to_csv(out_file, sep="\t")
    print(df)


def make_mem_dataframe(datasets: list[pathlib.Path], out_file: str):
    apps = ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]
    variations = ["Full", "Tri", "Packed"]
    # methods = [f"{app} {var}" for app in apps for var in variations]
    ds_names = [ds.name for ds in datasets]
    idx = pd.MultiIndex.from_product((ds_names, apps, variations))
    col_labels = ["rss", "vsize"]
    data = np.zeros((len(idx), len(col_labels)), dtype=np.int32)
    df = pd.DataFrame(data=data, columns=col_labels, index=idx)
    df.to_csv(out_file, sep="\t")
    print(df)


datasets.sort(key=sorter)
# make_dataframe(datasets, runtime_results_file)
# make_mem_dataframe(datasets, mem_results_file)

with tqdm(datasets) as t:
    # runtime_df = pd.read_csv(runtime_results_file, sep="\t", index_col=[0, 1])
    mem_df = pd.read_csv(mem_results_file, sep="\t", index_col=[0, 1, 2])
    for dataset in t:
        t.write(f"Running {dataset.name}")
        with open(dataset, "r") as f:
            d, k, n = [int(x) for x in f.readline().split()]
        template_args = {
            "n": n,
            "k": k,
            "d": d,
            "tri_size": d * (d - 1) // 2,
            "data_file": str(dataset),
        }
        with new_context() as ctx:
            project = get_packed_project(template_args)
            binary = (
                f"{project.output_path}/bin/{project.phases['Link Executable'].output}"
            )

            ninja_file = NinjaFile(project)
            ninja_file.generate()
            check_call(["ninja", "-f", ninja_file.path], stdout=DEVNULL)
            stdout, memdict = run_with_memory(binary, ".secretpass")
            t.write(stdout)
            t.write(str(memdict))
            mem_df.loc[dataset.name, "LAGrad", "Packed"] = (
                memdict.max_rss,
                memdict.max_vsize,
            )
            mem_df.to_csv(mem_results_file, sep="\t")
            # stdout = check_output([binary]).decode("utf-8")
            # lines = stdout.splitlines()[1:]
            # for line in lines:
            #     t.write(line)
            #     key, results = line.split(":")
            #     runtime_df.loc[dataset.name, key] = json.loads(results)
    # runtime_df.to_csv(runtime_results_file, sep="\t")
