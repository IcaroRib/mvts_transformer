import os
from itertools import product

if __name__ == "__main__":

    cities = {
        "manaus": 82332,
        "teresina": 82579,
        "salvador": 83248,
        "rio_janeiro": 83755,
        "cuiaba": 83362,
        "brasilia": 83378,
        "belo_horizonte": 83587,
        "curitiba": 83842,
        "porto_alegre": 83967,
    }

    datasets = [city + "_" + str(_id) for city, _id in cities.items()]
    learning_rates = [0.001]
    epochs = [40]
    param_list = product(datasets, learning_rates, epochs)

    for params in param_list:
        dataset = params[0]
        learning_rate = params[1]
        epoch = params[2]

        fullpath = f'./experiments/{dataset}/'
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)

        os.system(f'python main.py --output_dir experiments/{dataset}/ '
                  f'--comment "regression for {dataset}" '
                  f'--name {dataset}_Regression '
                  f'--records_file experiments/{dataset}/{dataset}_Regression.xls '
                  f'--data_dir datasets/files/{dataset}/ '
                  '--data_class wf '
                  '--pattern TRAIN '
                  '--val_pattern TEST '
                  f'--epochs {epoch} '
                  f'--lr {learning_rate} '
                  '--optimizer RAdam '
                  '--pos_encoding learnable '
                  '--task regression')

