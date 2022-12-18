import os

import sktime
from datasets import utils


if __name__ == '__main__':
    DATA_PATH = "C:/Users/iflr/PycharmProjects/mvts_transformer/src/datasets/files/BeijingPM25Quality_TEST.ts"

    df, labels = utils.load_from_tsfile_to_dataframe(DATA_PATH, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
    print(df)