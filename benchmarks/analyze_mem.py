import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
sns.set()

rcParams["font.family"] = "Times New Roman"


def ba_main():
    filepat = re.compile(r"ba\d+_n(?P<n>\d+)_m(?P<m>\d+)_p(?P<p>\d+).txt")
    df = pd.read_csv(
        "detailed_results/ba_memusage.csv", header=[0, 1, 2], index_col=0, skiprows=1
    )
    # There's a memory leak that causes the consumption to go up with each run, so we just take the first run.
    df = df[:1].unstack()
    df = df[:, :, "rss", "run1"] / 1e6
    sizes = df.index.get_level_values(0).str.extract(filepat).apply(pd.to_numeric)
    sizes["jacobian_cols"] = 11 * sizes.n + 3 * sizes.m + sizes.p
    idx = pd.MultiIndex.from_arrays(
        [sizes["jacobian_cols"], df.index.get_level_values(1)]
    )
    df.index = idx
    df.sort_index(level=[0, 1], inplace=True)
    print(df.index)

    fig, ax = plt.subplots(1, 1)
    for key in ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]:
        series = df[:, key]
        ax.plot(series.index, series, label=key, marker=".")
    ax.legend()
    ax.set_title("Memory consumption of BA")
    ax.set_xlabel("Number of columns in Jacobian")
    ax.set_ylabel("Peak resident set size (MB)")
    ax.set_yscale("log")

    plt.show()


def hand_main():
    filepat = re.compile(r"hand\d+_t\d+_c(?P<npts>\d+).txt")
    df = pd.read_csv(
        "detailed_results/hand_memusage.csv", header=[0, 1, 2], index_col=0, skiprows=1
    )
    # LAGrad doesn't memory leak but Enzyme/MLIR does. Take the first run again.
    df = df[:1].unstack()
    df = df[:, :, "rss", "run1"] / 1e6
    sizes = df.index.get_level_values(0).str.extract(filepat).apply(pd.to_numeric)
    idx = pd.MultiIndex.from_arrays([sizes["npts"], df.index.get_level_values(1)])
    df.index = idx
    fig, ax = plt.subplots(1, 1)
    for key in ["LAGrad", "Enzyme/MLIR"]:
        series = df[:, key]
        ax.plot(series.index, series, label=key, marker=".")
    ax.legend()
    ax.set_title("Memory consumption of Hand Tracking")
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Peak resident set size (MB)")
    ax.set_yscale("log")
    plt.show()


def lstm_main():
    mem_df = pd.read_csv(
        "detailed_results/lstm_memusage.tsv",
        sep="\t",
        index_col=[0, 1],
    )
    filepat = re.compile(r"lstm_l(?P<l>\d+)_c(?P<c>\d+).txt")
    sizes = mem_df.index.get_level_values(0).str.extract(filepat).apply(pd.to_numeric)
    sizes["jacobian"] = sizes.l * 8 * sizes.c
    idx = pd.MultiIndex.from_arrays(
        [sizes["jacobian"], mem_df.index.get_level_values(1)]
    )
    mem_df.index = idx
    print(mem_df)
    fig, ax = plt.subplots(1, 1)
    for key in ["LAGrad", "Enzyme/MLIR", "Enzyme/C", "Handrolled"]:
        series = mem_df["max_rss"][:, key] / 1e6
        ax.plot(series.index, series, label=key, marker=".")
    ax.legend()
    ax.set_title("Memory consumption of LSTM")
    ax.set_xlabel("Number of independent variables x sequence length")
    ax.set_ylabel("Peak resident set size (MB)")
    ax.set_yscale("log")
    plt.show()


def main_term_main():
    mem_df = pd.read_csv(
        "detailed_results/main_term_memusage.tsv",
        sep="\t",
        index_col=[0, 1],
        # skiprows=1,
    )
    # Ignore vsize for now.
    filepat = re.compile(r"gmm_d(?P<d>\d+)_K(?P<k>\d+).txt")
    sizes = mem_df.index.get_level_values(0).str.extract(filepat).apply(pd.to_numeric)
    sizes["jacobian"] = sizes.k + sizes.d * sizes.k * 2 + sizes.k * sizes.d ** 2
    idx = pd.MultiIndex.from_arrays(
        [sizes["jacobian"], mem_df.index.get_level_values(1)]
    )
    mem_df.index = idx

    fig, ax = plt.subplots(1, 1)
    for key in ["LAGrad", "Enzyme/C", "Handwritten"]:
        series = mem_df["max_rss"][:, key] / 1e6
        ax.plot(series.index, series, label=key, marker=".")
    ax.legend()
    ax.set_title("Memory consumption of Main Term (GMM subset)")
    ax.set_xlabel("Number of independent variables")
    ax.set_ylabel("Peak resident set size (MB)")
    ax.set_yscale("log")
    plt.show()


def matmul_main():
    mem_df = pd.read_csv(
        "detailed_results/matmul_memusage.tsv", sep="\t", index_col=[0, 1]
    )
    print(mem_df)
    fig, ax = plt.subplots(1, 1)
    for key in ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]:
        series = mem_df["max_rss"][:, key] / 1e6
        ax.plot(series.index, series, label=key, marker=".")
    ax.legend()
    ax.set_title("Memory consumption of Matrix Multiplication")
    ax.set_xlabel("Dimension of matrix")
    ax.set_ylabel("Peak resident set size (MB)")
    ax.set_yscale("log")
    plt.show()


if __name__ == "__main__":
    # main()
    # ba_main()
    lstm_main()
    # hand_main()
    # matmul_main()
