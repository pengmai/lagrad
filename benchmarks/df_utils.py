import pathlib
import pandas as pd
import numpy as np
from run_hand import get_datasets as get_hand_datasets


def main():
    # data_dir = (
    #     pathlib.Path.home()
    #     / "Research"
    #     / "Enzyme"
    #     / "enzyme"
    #     / "benchmarks"
    #     / "hand"
    #     / "data"
    #     / "simple_small"
    # )
    # datasets = get_hand_datasets(data_dir)
    # applications = ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]
    # midx = pd.MultiIndex.from_product((datasets, applications))
    # run_labels = [f"run{i + 1}" for i in range(6)]
    # data = np.zeros((len(midx), len(run_labels)))
    # df = pd.DataFrame(data=data, index=midx, columns=run_labels)
    # df.to_csv("detailed_results/hand_runtimes.tsv", sep="\t")

    df = pd.read_csv("detailed_results/hand_runtimes.tsv", sep="\t", index_col=[0, 1]).astype(np.int64)
    df.to_csv("detailed_results/hand_runtimes.tsv", sep="\t")
    # row = pd.Series(data=[23, 234, 24, 12, 12, 122], name="LAGrad")
    # # print(row.astype(np.float64))
    # df.loc["hand1_t26_c100.txt", "LAGrad"] = row.array
    # # print(df)
    # print(df.loc["hand1_t26_c100.txt", "LAGrad"])


if __name__ == "__main__":
    main()
