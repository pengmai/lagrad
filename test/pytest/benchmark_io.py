from dataclasses import dataclass, field
from typing import Tuple
import os
import numpy as np

DELIM = ":"


def parse_floats(arr):
    """Parses enumerable as float.
    Args:
        arr (enumerable): input data that can be parsed to floats.
    Returns:
        (List[float]): parsed data.
    """

    return [float(x) for x in arr]


@dataclass
class Wishart:
    gamma: np.float64 = 0.0
    m: np.int32 = 0


@dataclass
class GMMInput:
    alphas: np.ndarray = field(default=np.empty(0, dtype=np.float64))
    means: np.ndarray = field(default=np.empty(0, dtype=np.float64))
    icf: np.ndarray = field(default=np.empty(0, dtype=np.float64))
    x: np.ndarray = field(default=np.empty(0, dtype=np.float64))
    wishart: Wishart = field(default=Wishart())


def read_gmm_instance(fn, replicate_point):
    """Reads input data for GMM objective from the given file.
    Args:
        fn (str): input file name.
        replicate_point (bool): if False then file contains n different points,
            otherwise file contains only one point that will be replicated
            n times.

    Returns:
        (GMMInput): data for GMM objective test class.
    """
    with open(fn, "r") as fid:
        line = fid.readline()
        line = line.split()

        d = int(line[0])
        k = int(line[1])
        n = int(line[2])

        alphas = np.array([float(fid.readline()) for _ in range(k)])
        means = np.array([parse_floats(fid.readline().split()) for _ in range(k)])
        icf = np.array([parse_floats(fid.readline().split()) for _ in range(k)])

        if replicate_point:
            x_ = parse_floats(fid.readline().split())
            x = np.array([x_] * n)
        else:
            x = np.array([parse_floats(fid.readline().split()) for _ in range(n)])

        line = fid.readline().split()
        wishart_gamma = float(line[0])
        wishart_m = int(line[1])

    return GMMInput(alphas, means, icf, x, Wishart(wishart_gamma, wishart_m))


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
class HandModel:
    bone_count: int = field(default=0)

    bone_names: Tuple[str] = field(default=tuple())

    # asssuming that parent is earlier in the order of bones
    parents: np.ndarray = field(default=np.empty(0, dtype=np.int32))

    base_relatives: np.ndarray = field(default=np.empty(0, dtype=np.float64))

    inverse_base_absolutes: np.ndarray = field(default=np.empty(0, dtype=np.float64))

    base_positions: np.ndarray = field(default=np.empty(0, dtype=np.float64))

    weights: np.ndarray = field(default=np.empty(0, dtype=np.float64))

    # two dimensional array with the second dimension equals to 3
    triangles: np.ndarray = field(default=np.empty(0, dtype=np.int32))

    is_mirrored: bool = field(default=False)


@dataclass
class HandData:
    model: HandModel = field(default=HandModel())

    correspondences: np.ndarray = field(default=np.empty(0, dtype=np.int32))

    points: np.ndarray = field(default=np.empty(0, dtype=np.float64))


@dataclass
class HandInput:
    theta: np.ndarray = field(default=np.empty(0, dtype=np.float64))
    data: HandData = field(default=HandData())
    us: np.ndarray = field(default=np.empty(0, dtype=np.float64))


def load_model(path):
    """Loads HandModel from the given file.
    Args:
        path(str): path to a directory with input files.
    Returns:
        (HandModel): hand trcking model.
    """

    # Read in triangle info.
    triangles = np.loadtxt(os.path.join(path, "triangles.txt"), int, delimiter=DELIM)

    # Process bones file.
    bones_path = os.path.join(path, "bones.txt")

    # Grab bone names.
    bone_names = tuple(line.split(DELIM)[0] for line in open(bones_path))

    # Grab bone parent indices.
    parents = np.loadtxt(bones_path, int, usecols=[1], delimiter=DELIM).flatten()

    # Grab relative transforms.
    relative_transforms = np.loadtxt(
        bones_path, usecols=range(2, 2 + 16), delimiter=DELIM
    ).reshape(len(parents), 4, 4)

    vertices_path = os.path.join(path, "vertices.txt")
    n_bones = len(bone_names)

    # Find number of vertices.
    with open(vertices_path) as handle:
        n_verts = len(handle.readlines())

    # Read in vertex info.
    positions = np.zeros((n_verts, 3))
    weights = np.zeros((n_verts, n_bones))

    with open(vertices_path) as handle:
        for i_vert, line in enumerate(handle):
            atoms = line.split(DELIM)
            positions[i_vert] = parse_floats(atoms[:3])

            for i in range(int(atoms[8])):
                i_bone = int(atoms[9 + i * 2])
                weights[i_vert, i_bone] = float(atoms[9 + i * 2 + 1])

    # Grab absolute invers transforms.
    inverse_absolute_transforms = np.loadtxt(
        bones_path, usecols=range(2 + 16, 2 + 16 + 16), delimiter=DELIM
    ).reshape(len(parents), 4, 4)

    n_vertices = positions.shape[0]
    homogeneous_base_positions = np.ones((n_vertices, 4))
    homogeneous_base_positions[:, :3] = positions

    result = HandModel(
        n_bones,
        bone_names,
        parents,
        relative_transforms,
        inverse_absolute_transforms,
        homogeneous_base_positions,
        weights,
        triangles,
        False,  # WARNING: not exactly understand where such info comes from
    )

    return result


def read_hand_instance(model_dir, fn, read_us):
    """Reads input data for hand tracking objective.
    Args:
        model_dir (str): path to the directory contatins model data files.
        fn (str): name of the file contains additional data for objective.
        read_us (bool): if True then complicated scheme is used.
    Returns:
        (HandInput): input data for hand objective test class.
    """

    model = load_model(model_dir)

    with open(fn, "r") as fid:
        line = fid.readline()
        line = line.split()
        npts = int(line[0])
        ntheta = int(line[1])

        lines = [fid.readline().split() for _ in range(npts)]
        correspondences = np.array([int(line[0]) for line in lines])
        points = np.array([parse_floats(line[1:]) for line in lines])

        if read_us:
            us = np.array([parse_floats(fid.readline().split()) for _ in range(npts)])

        params = np.array([float(fid.readline()) for _ in range(ntheta)])

    data = HandData(model, correspondences, points)

    if read_us:
        return HandInput(params, data, us)
    else:
        return HandInput(params, data)


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
