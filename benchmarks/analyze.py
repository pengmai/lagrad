import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math

sns.set()
rcParams["font.family"] = "Times New Roman"


def plot_ba_absolute_runtimes(means, deviations, ax):
    for key in ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]:
        ax.errorbar(means["jacobian_cols"], means[key], deviations[key], label=key)

    ax.set_title("Absolute runtimes for BA: LAGrad vs Enzyme (MLIR and C)")
    ax.set_xlabel("Number of columns in Jacobian")
    ax.set_ylabel("Runtime (s)")
    ax.legend()


def plot_hand_absolute_runtimes(means, deviations, ax):
    for key in ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]:
        ax.errorbar(means["npts"], means[key], deviations[key], label=key)
    ax.set_title("Absolute Runtimes for Hand Tracking: LAGrad vs Enzyme (MLIR and C)")
    ax.set_ylabel("Runtime (s)")
    ax.set_xlabel("Number of observations")
    ax.legend()


def plot_lstm_absolute_runtimes(means, deviations, ax):
    for key in ["LAGrad", "Enzyme/MLIR", "Enzyme/C", "Handrolled"]:
        ax.errorbar(means["sizes"], means[key], deviations[key], label=key)
    ax.set_title("Absolute Runtimes for LSTMs: LAGrad vs Enzyme (MLIR and C)")
    ax.set_ylabel("Runtime (s)")
    ax.set_xlabel("Number of independent variables x sequence length")
    ax.legend()


def plot_gmm_absolute_runtimes(means, deviations, ax, title=""):
    for key in ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]:
        ax.errorbar(means["sizes"], means[key], deviations[key], label=key)

    # ax.set_title(
    #     "Absolute Runtimes for Gaussian Mixture Models: LAGrad vs Enzyme (MLIR and C)"
    # )
    ax.set_title(title)
    ax.set_ylabel("Runtime (s)")
    ax.set_xlabel("Number of independent variables")
    ax.legend()


def plot_relative_speedup(means):
    plt.plot(means["jacobian_cols"], means["Enzyme"] / means["LAGrad"])
    plt.ylim(bottom=0)

    plt.title("Relative BA speedup for LAGrad over Enzyme")
    plt.xlabel("# of columns in Jacobian")
    plt.ylabel("Speedup")
    # plt.legend()
    plt.show()


def process_ba(results_file):
    filepat = re.compile(r"ba\d+_n(?P<n>\d+)_m(?P<m>\d+)_p(?P<p>\d+).txt")
    df = pd.read_csv(results_file, header=[0, 1], index_col=0, skiprows=1)
    df = df[1:]  # Discard the first row as a warmup run
    means = df.mean().unstack() / 1e6
    deviations = df.std().unstack() / 1e6
    sizes = means.index.str.extract(filepat).apply(pd.to_numeric)
    sizes.index = means.index
    means["jacobian_cols"] = 11 * sizes.n + 3 * sizes.m + sizes.p
    deviations["jacobian_cols"] = means["jacobian_cols"]
    means, deviations = means.sort_values("jacobian_cols"), deviations.sort_values(
        "jacobian_cols"
    )
    # The last dataset is kind of messy to look at.
    means, deviations = means[:-1], deviations[:-1]
    return means, deviations


def process_hand(results_file):
    filepat = re.compile(r"hand\d+_t\d+_c(?P<npts>\d+).txt")
    df = pd.read_csv(results_file, header=[0, 1], index_col=0, skiprows=1)
    df = df[1:]  # Discard the first row as a warmup run
    means = df.mean().unstack() / 1e6
    deviations = df.std().unstack() / 1e6
    sizes = means.index.str.extract(filepat).apply(pd.to_numeric)
    sizes.index = means.index
    means["npts"] = sizes
    deviations["npts"] = means["npts"]
    means, deviations = means.sort_values("npts"), deviations.sort_values("npts")
    return means, deviations


def process_gmm(results_file):
    filepat = re.compile(r"gmm_d(?P<d>\d+)_K(?P<k>\d+).txt")
    df = pd.read_csv(results_file, header=[0, 1], index_col=0, skiprows=1)
    df = df[1:]  # Discard first row
    means = df.mean().unstack() / 1e6
    deviations = df.std().unstack() / 1e6
    # means, deviations = means.dropna(), deviations.dropna()
    sizes = means.index.str.extract(filepat).apply(pd.to_numeric)
    sizes.index = means.index
    means["sizes"] = sizes.k + sizes.d * sizes.k * 2 + sizes.k * sizes.d ** 2
    deviations["sizes"] = means["sizes"]
    means, deviations = means.sort_values("sizes"), deviations.sort_values("sizes")
    return means, deviations


def process_lstm(results_file):
    filepat = re.compile(r"lstm_l(?P<l>\d+)_c(?P<c>\d+).txt")
    df = pd.read_csv(results_file, header=[0, 1], index_col=0, skiprows=1)
    df = df[1:]
    means = df.mean().unstack() / 1e6
    deviations = df.std().unstack() / 1e6
    sizes = means.index.str.extract(filepat).apply(pd.to_numeric)
    sizes.index = means.index
    means["sizes"] = sizes.l * 8 * sizes.c
    deviations["sizes"] = means["sizes"]
    means, deviations = means.sort_values("sizes"), deviations.sort_values("sizes")
    return means, deviations


def smarter_round(sig):
    def rounder(x):
        return "{:.3f}".format(x)
        # Should I be using significant digits?
        """Modified from https://stackoverflow.com/a/57590725"""
        offset = sig - math.floor(math.log10(abs(x)))
        initial_result = round(x, offset)
        if str(initial_result)[-1] == "5" and initial_result == x:
            return "{:.3f}".format(round(x, offset - 2))
        else:
            return "{:.3f}".format(round(x, offset - 1))

    return rounder


def ba_results_to_latex(means, stds):
    means["Speedup"] = (means["Enzyme"] / means["LAGrad"]).apply(
        lambda x: "{:.2f}".format(x)
    )
    for key in ["LAGrad", "Enzyme"]:
        means[key] = (
            "$"
            + means[key].apply(smarter_round(3))
            + " \pm "
            + stds[key].apply(smarter_round(3))
            + "$"
        )
    tablestr = means[["jacobian_cols", "LAGrad", "Enzyme", "Speedup"]].to_csv(
        index=False, header=False
    )
    tablestr = tablestr.replace(",", " & ")
    tablestr = tablestr.replace("\n", "\\\\\n\\hline\n")
    print(tablestr)


def hand_results_to_latex(means: pd.DataFrame, stds):
    means["Speedup"] = (means["Enzyme/MLIR"] / means["LAGrad"]).apply(
        lambda x: "{:.2f}".format(x)
    )
    for key in ["LAGrad", "Enzyme/MLIR", "Enzyme/C"]:
        means[key + "_table"] = (
            "$"
            + means[key].apply(smarter_round(3))
            + " \pm "
            + stds[key].apply(smarter_round(3))
            + "$"
        )
        means.loc[means[key].isnull(), key + "_table"] = ""
        means[key] = means[key + "_table"]
    tablestr = means[["npts", "LAGrad", "Enzyme/MLIR", "Enzyme/C", "Speedup"]].to_csv(
        index=False, header=False
    )
    tablestr = tablestr.replace(",", " & ")
    tablestr = tablestr.replace("\n", "\\\\\n\\hline\n")
    print(tablestr)


def plot_gmm_results(tri_results_file, full_results_file):
    gmeans, gdev = process_gmm(tri_results_file)
    gfullmeans, gfulldev = process_gmm(full_results_file)
    # gcompmeans, gcompdev = process_gmm(GMM_COMP_RESULTS_FILE)
    # gcompmeans["LAGrad"] = np.nan
    # gcompdev["LAGrad"] = np.nan
    fig, ax = plt.subplots(1, 1)
    for key, color in zip(["LAGrad", "Enzyme/MLIR", "Enzyme/C"], ["C0", "C1", "C2"]):
        ax.errorbar(
            gmeans["sizes"],
            gmeans[key],
            gdev[key],
            label=f"{key} Triangular",
            ls="dashed",
            color=color,
        )
        ax.errorbar(
            gfullmeans["sizes"],
            gfullmeans[key],
            gfulldev[key],
            label=f"{key} Full",
            color=color,
        )

    # ax.set_title(
    #     "Absolute Runtimes for Gaussian Mixture Models: LAGrad vs Enzyme (MLIR and C)"
    # )
    # ax.set_title(title)
    ax.set_ylabel("Runtime (s)")
    ax.set_xlabel("Number of independent variables")
    ax.legend()

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # plot_gmm_absolute_runtimes(gmeans, gdev, ax1, title="Triangular computation, full materialization")
    # plot_gmm_absolute_runtimes(
    #     gfullmeans, gfulldev, ax2, title="Full computation, full materialization"
    # )
    fig.suptitle("Absolute runtimes for Gaussian Mixture Models")
    plt.show()


if __name__ == "__main__":
    BA_RESULTS_FILE = "detailed_results/ba_detailed_results.csv"
    HAND_RESULTS_FILE = "detailed_results/hand_detailed_results.csv"
    GMM_TRI_RESULTS_FILE = "detailed_results/gmm_tri_detailed_results.csv"
    GMM_FULL_RESULTS_FILE = "detailed_results/gmm_full_detailed_results.csv"
    GMM_COMP_RESULTS_FILE = "detailed_results/gmm_detailed_results.csv"
    LSTM_RESULTS_FILE = "detailed_results/lstm_detailed_results.csv"
    bmeans, bdev = process_ba(BA_RESULTS_FILE)
    # print(tablestr)
    # print(bmeans, bdev)
    hmeans, hdev = process_hand(HAND_RESULTS_FILE)
    lmeans, ldev = process_lstm(LSTM_RESULTS_FILE)

    fix, ax = plt.subplots(1, 1)
    plot_lstm_absolute_runtimes(lmeans, ldev, ax)
    plt.show()
    # hand_results_to_latex(hmeans, hdev)

    # plot_gmm_results(GMM_TRI_RESULTS_FILE, GMM_FULL_RESULTS_FILE)

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # plot_ba_absolute_runtimes(bmeans, bdev, ax1)
    # plot_hand_absolute_runtimes(hmeans, hdev, ax2)
    # plt.show()
