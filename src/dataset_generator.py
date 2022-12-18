import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def read_dataset(path):

    columns = ['date', 'hour', 'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco']
    dataframe = pd.read_csv(path, names=columns)
    dataframe = dataframe.loc[dataframe['date'] >= '2015-12-23']
    X_dataframe = dataframe.drop(columns=['wpgt', 'snow', 'tsun', 'coco'], axis=1)
    X_dataframe.reset_index()
    y_dataframe = X_dataframe.pop('temp')

    total_days = len(X_dataframe['date'].unique())
    cut = round(total_days/ 4 * 3)
    count = 0
    last_date = None
    last_index = 0

    for index, row in X_dataframe.iterrows():
        date = row['date']
        if date != last_date:
            last_date = date
            count += 1
        if count > cut:
            last_index = index
            break

    X_train = X_dataframe.loc[:last_index]
    X_test = X_dataframe.loc[last_index:]
    y_train = y_dataframe.loc[:last_index]
    y_test = y_dataframe.loc[last_index:]

    return X_train, X_test, y_train, y_test


def clean_data(X_train, X_test, feature_scaling = False):

    train_time_column = X_train.pop("date")
    test_time_column = X_test.pop("date")

    train_hour_column = X_train.pop("hour")
    test_hour_column = X_test.pop("hour")

    for column in X_train.columns:
        if column == 'prcp':
            X_train[column] = X_train[column].fillna(0)
            X_test[column] = X_test[column].fillna(0)
        else:
            mean = round(X_train[column].mean(), 4)
            X_train[column] = X_train[column].fillna(mean)
            X_test[column] = X_test[column].fillna(mean)

    if feature_scaling:
        min_max_scaler = MinMaxScaler()
        x_scaled = pd.DataFrame(min_max_scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        x_test_scaled = pd.DataFrame(min_max_scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

        X_train = x_scaled.round(4)
        X_test = x_test_scaled.round(4)

    X_train["date"] = train_time_column
    X_test["date"] = test_time_column

    X_train["hour"] = train_hour_column
    X_test["hour"] = test_hour_column

    return X_train, X_test


def format_file(train_df, y_train, test_df, y_test):
    train_instances = []
    test_instances = []
    train_df['tavg'] = y_train
    test_df['tavg'] = y_test

    columns = ['dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres']
    dates = train_df['date'].unique()
    for date in dates:
        instance = []
        date_df = train_df.loc[train_df['date'] == date]
        tavg = round(date_df['tavg'].mean(), 4)
        for column in columns:
            dimension = []
            i = 0
            mean = round(date_df[column].mean(), 4)
            for index, row in date_df.iterrows():
                value = row[column]
                hour = row['hour']
                while i < hour:
                    dimension.append(mean)
                    i += 1
                dimension.append(value)
                i += 1
            while i < 24:
                dimension.append(mean)
                i += 1
            instance.append(dimension)
        instance.append(tavg)
        train_instances.append(instance)

    dates = test_df['date'].unique()
    for date in dates:
        instance = []
        date_df = test_df.loc[test_df['date'] == date]
        tavg = round(date_df['tavg'].mean(), 4)
        for column in columns:
            dimension = []
            i = 0
            mean = round(date_df[column].mean(), 4)
            for index, row in date_df.iterrows():
                value = row[column]
                hour = row['hour']
                while i < hour:
                    dimension.append(mean)
                    i += 1
                dimension.append(value)
                i += 1
            while i < 24:
                dimension.append(mean)
                i += 1
            instance.append(dimension)
        instance.append(tavg)
        test_instances.append(instance)

    return train_instances, test_instances

def write_ts_file(path, file, train_instances, test_instances):
    header = """
@problemname {file}
@timestamps false
@missing true
@univariate false
@dimension 6
@equallength true
@serieslength 24
@targetlabel true
@data    
"""

    with open(path+file+'_TRAIN.ts','w') as ts_file:
        ts_file.write(header.format(file=file))
        for instance in train_instances:
            line = ":".join(map(str, instance)) + "\n"
            line = line.replace("'","")
            line = line.replace(" ", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace("$", " ")
            ts_file.write(line)

    with open(path+file+'_TEST.ts','w') as ts_file:
        ts_file.write(header.format(file=file))
        for instance in test_instances:
            line = ":".join(map(str, instance)) + "\n"
            line = line.replace("'", "")
            line = line.replace(" ", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace("$", " ")
            ts_file.write(line)


if __name__ == "__main__":
    path = 'datasets/files/salvador_83248/'
    csv_file = '83248.csv'
    ts_file = 'Salvador83248'

    X_train, X_test, y_train, y_test = read_dataset(path+csv_file)
    X_train, X_test = clean_data(X_train, X_test)
    train_instances, test_instances = format_file(X_train, y_train, X_test, y_test)
    write_ts_file(path, ts_file, train_instances, test_instances)

