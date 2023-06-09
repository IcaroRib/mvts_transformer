import os
from itertools import product

if __name__ == "__main__":
    #
    # cities = {
    #     "manaus": 82332,
    #     "teresina": 82579,
    #     "salvador": 83248,
    #     "rio_janeiro": 83755,
    #     "cuiaba": 83362,
    #     "brasilia": 83378,
    #     "belo_horizonte": 83587,
    #     "curitiba": 83842,
    #     "porto_alegre": 83967,
    # }

    cities = {
        "manaus": "a101",
        "teresina": "a312",
        "rio_janeiro": "a652",
        "brasilia": "a001",
        "curitiba": "a807",
    }

    datasets = [city + "_" + str(_id) for city, _id in cities.items()]
    learning_rates = [0.001, 0.0025, 0.005]
    epochs = [30]
    param_list = product(datasets, learning_rates, epochs)
    target_name = 'prep_tot'

    for params in param_list:
        dataset = params[0]
        learning_rate = params[1]
        epoch = params[2]

        fullpath = f'./experiments/{dataset}/{target_name}/'
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)

        os.system(f'python main.py --output_dir experiments/{dataset}/{target_name} '
                  f'--comment "regression for {dataset}" '
                  f'--name {dataset}_Regression '
                  f'--records_file experiments/{dataset}/{target_name}/{dataset}_Regression.xls '
                  f'--data_dir datasets/files_v2/{dataset}/{target_name}/ '
                  '--data_class wf '
                  '--pattern TRAIN '
                  '--val_pattern TEST '
                  f'--epochs {epoch} '
                  f'--lr {learning_rate} '
                  '--optimizer Adam '
                  '--pos_encoding learnable '
                  '--task regression')

