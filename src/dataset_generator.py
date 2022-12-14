import pandas as pd
import numpy as np

HEADER = """
@problemname Curitiba83842
@timestamps false
@missing true
@univariate false
@dimension 7
@equallength true
@serieslength 1
@targetlabel true
@data
"""

if __name__ == "__main__":
    path = './datasets/files/curitiba_83842/'
    columns = ['date','tavg','tmin','tmax','prcp','snow','wdir','wspd','wpgt','pres','tsun']
    dataframe = pd.read_csv(path+'83842.csv', names=columns)
    columns.remove('wpgt')
    columns.remove('snow')
    columns.remove('tsun')
    columns.remove('tavg')
    cut = round(len(dataframe) - len(dataframe) / 3)
    train_instances = []
    test_instances = []

    for index, row in dataframe.iterrows():
        instance = []
        time = row['date']
        tavg = row['tavg']
        for column in range(1, len(columns)):
            value = row[column]
            if not pd.isnull(value):
                dimension = (time, value)
            instance.append(dimension)
        instance.append(tavg)
        if index < cut:
            train_instances.append(instance)
        else:
            test_instances.append(instance)

    with open(path+'83842_TRAIN.ts','w') as ts_file:
        ts_file.write(HEADER)
        for instance in train_instances:
            line = ":".join(map(str, instance)) + "\n"
            line = line.replace("'","")
            line = line.replace(" ", "")
            ts_file.write(line)

    with open(path+'83842_TEST.ts','w') as ts_file:
        ts_file.write(HEADER)
        for instance in test_instances:
            line = ":".join(map(str, instance)) + "\n"
            line = line.replace("'", "")
            line = line.replace(" ", "")
            ts_file.write(line)

    #
    #
    # print(dataframe)