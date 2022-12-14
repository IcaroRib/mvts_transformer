import os

import sktime
from sktime.datasets import load


if __name__ == '__main__':
    DATA_PATH = "C:/Users/iflr/PycharmProjects/mvts_transformer/src/datasets/files/"

    train_x, train_y = load_from_tsfile_to_dataframe(
        os.path.join(DATA_PATH, "teste.ts")
    )