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
