from dataclasses import dataclass, field

import numpy as np


def parse_floats(arr):
    """Parses enumerable as float.
    Args:
        arr (enumerable): input data that can be parsed to floats.
    Returns:
        (List[float]): parsed data.
    """

    return [float(x) for x in arr]


@dataclass
class BAInput:
    cams: np.ndarray = field(default=np.empty(0, dtype=np.float64))
    x: np.ndarray = field(default=np.empty(0, dtype=np.float64))
    w: np.ndarray = field(default=np.empty(0, dtype=np.float64))
    obs: np.ndarray = field(default=np.empty(0, dtype=np.int32))
    feats: np.ndarray = field(default=np.empty(0, dtype=np.float64))


def read_ba_instance(fn):
    """Reads input data for BA objective from the given file.
    Args:
        fn (str): input file name.
    Returns:
        (BAInput): input data for BA objective test class.
    """

    fid = open(fn, "r")
    with open(fn, "r") as fid:
        line = fid.readline()
        line = line.split()

        n = int(line[0])
        m = int(line[1])
        p = int(line[2])

        one_cam = parse_floats(fid.readline().split())
        cams = np.tile(one_cam, (n, 1))

        one_X = parse_floats(fid.readline().split())
        X = np.tile(one_X, (m, 1))

        one_w = float(fid.readline())
        w = np.tile(one_w, p)

        one_feat = parse_floats(fid.readline().split())
        feats = np.tile(one_feat, (p, 1))

    camIdx = 0
    ptIdx = 0
    obs = []
    for _ in range(p):
        obs.append((camIdx, ptIdx))
        camIdx = (camIdx + 1) % n
        ptIdx = (ptIdx + 1) % m

    obs = np.array(obs)

    return BAInput(cams, X, w, obs, feats)


@dataclass
class LSTMInput:
    main_params: np.ndarray = field(default=np.empty(0, dtype=np.float64))
    extra_params: np.ndarray = field(default=np.empty(0, dtype=np.float64))
    state: np.ndarray = field(default=np.empty(0, dtype=np.float64))
    sequence: np.ndarray = field(default=np.empty(0, dtype=np.float64))


def read_lstm_instance(fn):
    """Reads input data for LSTM objective from the given file.
    Args:
        fn (str): input file name.
    Returns:
        (LSTMInput): input data for LSTM objective test class.
    """
    with open(fn) as fid:
        line = fid.readline().split()
        layer_count = int(line[0])
        char_count = int(line[1])
        # char_bits = int(line[2])

        fid.readline()
        main_params = np.array(
            [parse_floats(fid.readline().split()) for _ in range(2 * layer_count)]
        )

        fid.readline()
        extra_params = np.array(
            [parse_floats(fid.readline().split()) for _ in range(3)]
        )

        fid.readline()
        state = np.array(
            [parse_floats(fid.readline().split()) for _ in range(2 * layer_count)]
        )

        fid.readline()
        text_mat = np.array(
            [parse_floats(fid.readline().split()) for _ in range(char_count)]
        )

    return LSTMInput(main_params, extra_params, state, text_mat)
