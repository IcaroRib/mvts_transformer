import os
from itertools import product

if __name__ == "__main__":

    #datasets = ['belo_horizonte_83587', 'manaus_82332', 'salvador_83248']
    #datasets = ['curitiba_83842', 'cuiaba_83362']
    datasets = ["brasilia_83378"]
    learning_rates = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01]
    epochs = [10, 20, 30, 40, 50]
    param_list = product(datasets, learning_rates, epochs)

    for params in param_list:
        dataset = params[0]
        learning_rate = params[1]
        epoch = params[2]

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

