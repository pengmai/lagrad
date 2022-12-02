import pathlib
import ronin.utils.messages as rum


def myannounce(message, prefix="rōnin", color="green"):
    pass


rum.announce = myannounce

from ronin.ninja import NinjaFile
from ronin.contexts import new_context
from subprocess import check_call, check_output, DEVNULL
import json
import pandas as pd
import re
from build import get_project

from tqdm import tqdm

runtime_results_file = (
    pathlib.Path.home()
    / "Research"
    / "lagrad"
    / "detailed_results"
    / "trmv_runtimes.tsv"
)


def make_dataframe(datasets, out_file):
    import numpy as np

    n_runs = 6
    methods = ["LAGrad Packed", "LAGrad Tri", "LAGrad Full"]
    copied_datasets = [ds for ds in datasets for _ in range(len(methods))]
    n_cols = len(copied_datasets)
    copied_methods = methods * len(datasets)
    # run_labels = [f"run{i+1}" for i in range(n_runs)]
    columns = ["rss", "vsize"]
    data = np.zeros(n_cols * n_runs, dtype=np.int32).reshape(n_cols, n_runs)
    idx = pd.MultiIndex.from_tuples(zip(copied_datasets, copied_methods))
    df = pd.DataFrame(data=data, columns=columns, index=idx)
    df.to_csv(out_file, sep="\t")
    print(df)


sizes = [256, 512, 1024, 2048, 4096]
make_dataframe(sizes, runtime_results_file)

with tqdm(sizes) as t:
    runtime_df = pd.read_csv(runtime_results_file, sep="\t", index_col=[0, 1])
    for n in t:
        t.write(f"Running {n}")
        template_args = {
            "n": n,
            "tri_size": n * (n - 1) // 2,
        }
        with new_context() as ctx:
            project = get_project(template_args)
            binary = (
                f"{project.output_path}/bin/{project.phases['Link Executable'].output}"
            )

            ninja_file = NinjaFile(project)
            ninja_file.generate()
            check_call(["ninja", "-f", ninja_file.path], stdout=DEVNULL)
            stdout = check_output([binary]).decode("utf-8")
            lines = stdout.splitlines()
            for line in lines:
                t.write(line)
                key, results = line.split(":")
                runtime_df.loc[n, key] = json.loads(results)
    runtime_df.to_csv(runtime_results_file, sep="\t")
