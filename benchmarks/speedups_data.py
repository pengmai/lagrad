import numpy as np
import pandas as pd


def get_trmv_data():
    # N = 1024, 2048, 4096
    lagrad_trmv = [
        [2688, 595, 575, 546, 544, 545],
        [15872, 2900, 2998, 3125, 2950, 2968],
        [44772, 17434, 16504, 16839, 16243, 16812],
    ]
    enzyme_trmv = [
        [6624, 6489, 4396, 4211, 4102, 4159],
        [27811, 28342, 21301, 21597, 20825, 20223],
        [133492, 104312, 99832, 96511, 100292, 96695],
    ]
    enzyme_c_trmv = [
        [4175, 6817, 4040, 4348, 4012, 3879],
        [18898, 22131, 16132, 16041, 13651, 14240],
        [105074, 116191, 97558, 92454, 90184, 90577],
    ]
    pytorch_trmv = [
        [2365, 2284, 1119, 1112, 1120, 1122],
        [5162, 10294, 5171, 5182, 5062, 5057],
        [18931, 44367, 29527, 29172, 29434, 29126],
    ]
    results = [
        np.median(np.array(x)[:, 1:], axis=1)
        for x in [lagrad_trmv, enzyme_trmv, enzyme_c_trmv, pytorch_trmv]
    ]
    results = np.vstack(results) / 1e6
    return results[1] / results

def get_gmm_data():
    gmm_data = pd.read_csv(
        "detailed_results/gmm_packed_runtimes.tsv", sep="\t", index_col=[0, 1]
    )
    gmm_data = gmm_data.loc[["gmm_d64_K10.txt", "gmm_d64_K50.txt", "gmm_d128_K200.txt"]]
    gmm_data = gmm_data.drop("run1", axis=1).median(axis=1) / 1e6
    gmm_data = gmm_data.unstack(0).__array__()[::-1]
    gmm_pytorch_data = [
        [2751955, 2491018, 2507231, 2438084, 2491156, 2733361],
        [4246824, 3919404, 3744089, 3741467, 3816284, 3674329],
        [17274496, 16982539, 17154168, 16991294, 17072565, 17715130],
    ]
    gmm_pytorch_data = np.median(np.array(gmm_pytorch_data)[:, 1:], axis=1) / 1e6
    gmm_data = np.vstack((gmm_data, gmm_pytorch_data))
    gmm_data = gmm_data[1] / gmm_data
    return gmm_data


def get_lstm_data():
    lstm_data = pd.read_csv(
        "detailed_results/lstm_runtimes.tsv", sep="\t", index_col=[0, 1]
    )
    lstm_data = lstm_data.loc[
        ["lstm_l2_c1024.txt", "lstm_l4_c1024.txt", "lstm_l4_c4096.txt"]
    ]
    lstm_data = lstm_data.drop("run1", axis=1).median(axis=1) / 1e6
    lstm_data = lstm_data.unstack(level=0).__array__()[::-1]

    pytorch_lstm_data = [
        [1181893, 1152040, 1143756, 1147507, 1147359, 1159702],
        [1864042, 1852132, 1843202, 1861533, 1897396, 1894279],
        [7808044, 7828103, 7767292, 7804919, 7803633, 7768893],
    ]
    pytorch_lstm_data = np.median(np.array(pytorch_lstm_data)[:, 1:], axis=1) / 1e6
    lstm_data = np.vstack((lstm_data, pytorch_lstm_data))
    lstm_data = lstm_data[1] / lstm_data
    return lstm_data


def get_hand_data():
    lagrad_hand_rt = [
        [54553, 53538, 53930, 52518, 54205, 56301],
        [1755258, 1744541, 1741825, 1730381, 1731462, 1769748],
        [37734573, 37350676, 37311152, 37216484, 37161792, 37195290],
    ]
    # These results are for hand11, not hand12, because I didn't collect results for Enzyme/MLIR hand12
    enzyme_hand_rt = [
        [143769, 143079, 142656, 146584, 148251, 147508],
        [4748687, 4755966, 4760999, 4763770, 4765746, 4761487],
        [125829848, 125720774, 126061361, 125884548, 126059980, 126860122],
    ]
    enzyme_c_hand_rt = [
        [2062957, 2082801, 2078484, 2140845, 2106993, 2122447],
        [73072396, 77361299, 73813498, 73515856, 74249215, 72549840],
        [np.nan] * 6,  # Timed out
    ]
    pytorch_hand_rt = [
        [3711702, 3688345, 3711433, 3688196, 3673347, 3661379],
        [725588808] * 6,
        [np.nan] * 6,  # Timed out
    ]
    results = [
        np.median(np.array(x)[:, 1:], axis=1)
        for x in [lagrad_hand_rt, enzyme_hand_rt, enzyme_c_hand_rt, pytorch_hand_rt]
    ]
    results = np.vstack(results) / 1e6
    return results[1] / results


def get_ba_data():
    lagrad_ba = [
        [39698, 32807, 32885, 34429, 33021, 33103],
        [658479, 576276, 580522, 578134, 578284, 579871],
        [10995755, 9595123, 9752538, 9516452, 9520890, 9659809],
    ]
    enzyme_mlir_ba = [
        [83081, 82497, 82201, 82368, 82415, 82130],
        [1477084, 1491221, 1488214, 1488721, 1465486, 1481065],
        [24641927, 24591707, 25907173, 25050231, 25693034, 26185084],
    ]
    enzyme_c_ba = [
        [86332, 81416, 83631, 82314, 80894, 81246],
        [1386639, 1339497, 1313908, 1325005, 1316472, 1311859],
        [25049568, 23482628, 23482635, 23519387, 23518336, 23698729],
    ]
    pytorch_ba = [
        [56913694, 56125022, 56162266, 56275437, 56259753, 56306846],
        [np.nan] * 6,  # timed out
        [np.nan] * 6,  # timed out
    ]
    results = [
        np.median(np.array(x)[:, 1:], axis=1)
        for x in (lagrad_ba, enzyme_mlir_ba, enzyme_c_ba, pytorch_ba)
    ]
    results = np.vstack(results) / 1e6
    return results[1] / results


def get_mlp_data():
    # hsize = 256, 512, 1024
    lagrad_mlp = [
        [1171, 765, 703, 694, 688, 739],
        [3167, 1690, 1660, 1949, 1813, 1629],
        [7420, 4564, 4521, 4601, 4533, 4528],
    ]
    pytorch_mlp = [
        [3901, 2239, 1783, 1691, 1740, 1821],
        [6819, 4795, 3973, 3982, 3675, 3609],
        [18792, 13465, 11428, 11249, 11213, 11154],
    ]
    enzyme_mlir_mlp = [
        [63671, 41640, 40282, 40711, 40475, 40263],
        [160006, 122171, 121603, 116068, 114626, 118098],
        [580158, 580282, 560725, 563894, 569117, 580856],
    ]
    enzyme_c_mlp = [
        [48473, 44002, 42912, 41160, 41265, 41307],
        [129731, 117548, 117301, 117034, 117005, 116655],
        [506891, 500793, 500953, 508381, 498882, 498636],
    ]

    results = [
        np.median(np.array(x)[:, 1:], axis=1)
        for x in (lagrad_mlp, enzyme_mlir_mlp, enzyme_c_mlp, pytorch_mlp)
    ]
    results = np.vstack(results) / 1e6
    return results[1] / results


def trmv_memory():
    lagrad_mem = [9236480, 34435072, 135_180_288]
    enzyme_mlir_mem = [26_087_424, 69390336, 254693376]
    enzyme_c_mem = [26099712, 69_079_040, 215957504]
    pytorch_mem = [119_287_808, 156_491_776, 312_582_144]
    results = np.vstack((lagrad_mem, enzyme_mlir_mem, enzyme_c_mem, pytorch_mem)) / 1e6
    return results[1] / results


def gmm_memory():
    lagrad_mem = [7_016_448, 10_620_928, 78_237_696]
    enzyme_mlir_mem = [110_305_280, 526_536_704, 4_190_023_680]
    enzyme_c_mem = [110_497_792, 527_355_904, 4_203_220_992]
    pytorch_mem = [368_406_528, 988_254_208, 6_388_604_928]
    results = np.vstack((lagrad_mem, enzyme_mlir_mem, enzyme_c_mem, pytorch_mem)) / 1e6
    return results[1] / results


def ba_memory():
    lagrad_mem = [16322560, 275181568, 4422504448]
    enzyme_mlir_mem = [15310848, 256827392, 4125827072]
    enzyme_c_mem = [14270464, 238538752, 3829125120]
    pytorch_mem = [1682735104] + [np.nan] * 2
    results = np.vstack((lagrad_mem, enzyme_mlir_mem, enzyme_c_mem, pytorch_mem)) / 1e6
    return results[1] / results


def lstm_memory():
    lagrad_mem = [4145152, 7163904, 26_075_136]
    enzyme_mlir_mem = [15273984, 27508736, 107_560_960]
    enzyme_c_mem = [5242880, 9277440, 28_172_288]
    pytorch_mem = [222654464, 278831104, 717_647_872]
    results = np.vstack((lagrad_mem, enzyme_mlir_mem, enzyme_c_mem, pytorch_mem)) / 1e6
    return results[1] / results


def hand_memory():
    lagrad_mem = [1679360, 3563520, 37515264]
    enzyme_mlir_mem = [4235264, 37326848, 518393856]
    enzyme_c_mem = [11763712, 21835776, np.nan]
    pytorch_mem = [136_421_376, 167_325_696, np.nan]  # 621_215_744
    results = np.vstack((lagrad_mem, enzyme_mlir_mem, enzyme_c_mem, pytorch_mem)) / 1e6
    return results[1] / results


def nn_memory():
    lagrad_mem = [6_774_784, 14_508_032, 31_178_752]
    enzyme_mlir_mem = [57_729_024, 115_941_376, 239_366_144]
    enzyme_c_mem = [5_775_360, 11_345_920, 28_102_656]
    pytorch_mem = [120_221_696, 125_005_824, 138_682_368]
    results = np.vstack((lagrad_mem, enzyme_mlir_mem, enzyme_c_mem, pytorch_mem)) / 1e6
    return results[1] / results

if __name__ == "__main__":
    data = trmv_memory()
    # Geomean w.r.t. Enzyme/MLIR
    from scipy import stats
    print(stats.gmean(data[0]))
    print(data)
