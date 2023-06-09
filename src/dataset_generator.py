import pandas as pd
import urllib.request
import gzip
import shutil
import os
from sklearn.preprocessing import MinMaxScaler


def read_dataset_v2(path, target_name = "temp"):
    columns = ['date', 'hour', 'prep_tot',
               'pres_atm', 'max_pres_atm', 'min_pres_atm',
               "rad_glob",
               "temp", "dew_point", "max_temp", "min_temp",
               "max_dew", "min_dew",
               "max_humi", "min_humi", "humi",
               "wind_direc", "max_wind", "wind_speed", "unnamed"]

    combined_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename == 'temp':
            continue
        fullpath = os.path.join(path, filename)
        df = pd.read_csv(fullpath, names=columns, header=8, encoding='latin-1', sep=';', dtype=str)

        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['hour'] = df['hour'].str[:2]
        df['hour'] = df['hour'].astype(int)
        for column in columns[2:]:
            df[column] = df[column].str.replace(",", ".")
            df[column] = df[column].astype(float)

        combined_df = pd.concat([combined_df, df], ignore_index=True)

    filtered_df = combined_df[(combined_df != -9999).all(axis=1)]
    X_dataframe = filtered_df.drop("unnamed", axis=1)
    print(f"Dataset {path} | Orifinal size {len(X_dataframe['date'].unique())}")
    X_dataframe.dropna(subset=[target_name], inplace=True)
    print(f"Dataset {path} | New size {len(X_dataframe['date'].unique())}")

    y_dataframe = X_dataframe[['date', 'hour', target_name]].copy()
    X_dataframe.pop(target_name)

    X_dates = X_dataframe['date'].unique()[:-1]
    y_dates = y_dataframe['date'].unique()[1:]

    cut = round(len(X_dates) / 4 * 3) + (len(X_dates) % 8)

    X_cutoff_date = X_dates[cut]
    X_end_date = X_dates[-1]

    y_start_date = y_dates[0]
    y_cutoff_date = y_dates[cut]

    X_train = X_dataframe.loc[X_dataframe['date'] <= X_cutoff_date]
    X_test = X_dataframe.loc[(X_dataframe['date'] > X_cutoff_date) & (X_dataframe['date'] <= X_end_date)]
    y_train = y_dataframe.loc[(y_dataframe['date'] >= y_start_date) & (y_dataframe['date'] <= y_cutoff_date)]
    y_test = y_dataframe.loc[y_dataframe['date'] > y_cutoff_date]

    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    y_train = y_train.reset_index()
    y_test = y_test.reset_index()

    return X_train, X_test, y_train, y_test

def read_dataset(path, target_name = 'temp'):

    columns = ['date', 'hour', 'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'coco']
    dataframe = pd.read_csv(path, names=columns)
    dataframe = dataframe.loc[(dataframe['date'] >= '2010-01-01')]
    X_dataframe = dataframe.drop(columns=['wpgt', 'snow', 'tsun', 'coco'], axis=1)

    print(f"Dataset {path} | Orifinal size {len(X_dataframe['date'].unique())}")
    X_dataframe.dropna(subset=[target_name], inplace=True)
    print(f"Dataset {path} | New size {len(X_dataframe['date'].unique())}")

    y_dataframe = X_dataframe[['date', 'hour', target_name]].copy()
    X_dataframe.pop(target_name)

    X_dates = X_dataframe['date'].unique()[:-1]
    y_dates = y_dataframe['date'].unique()[1:]

    cut = round(len(X_dates)/ 4 * 3) + (len(X_dates) % 8)

    X_cutoff_date = X_dates[cut]
    X_end_date = X_dates[-1]

    y_start_date = y_dates[0]
    y_cutoff_date = y_dates[cut]

    X_train = X_dataframe.loc[X_dataframe['date'] <= X_cutoff_date]
    X_test = X_dataframe.loc[(X_dataframe['date'] > X_cutoff_date) & (X_dataframe['date'] <= X_end_date)]
    y_train = y_dataframe.loc[(y_dataframe['date'] >= y_start_date) & (y_dataframe['date'] <= y_cutoff_date)]
    y_test = y_dataframe.loc[y_dataframe['date'] > y_cutoff_date]

    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    y_train = y_train.reset_index()
    y_test = y_test.reset_index()

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


def format_file(train_df, y_train, test_df, y_test, columns, target_name ='temp'):
    train_df[target_name] = y_train[target_name]
    test_df[target_name] = y_test[target_name]

    dates = train_df['date'].unique()
    train_instances = create_rows(target_name, columns, dates, train_df)
    dates = test_df['date'].unique()
    test_instances = create_rows(target_name, columns, dates, test_df)

    return train_instances, test_instances


def create_rows(target_name, columns, dates, df):
    instances = []
    for date in dates:
        instance = []
        date_df = df.loc[df['date'] == date]
        target_value = round(date_df[target_name].sum(), 4)

        for column in columns:
            dimension_list = ['x' for i in range(24)]
            mean = round(date_df[column].mean(), 4)

            for index, row in date_df.iterrows():
                value = row[column]
                hour = row['hour']
                dimension_list[hour] = value

            dimension_list = [mean if val == 'x' else val for val in dimension_list]
            instance.append(dimension_list)

        instance.append(target_value)
        instances.append(instance)
    return instances

def write_ts_file(path, file, train_instances, test_instances):
    header = """
@problemname {file}
@timestamps false
@missing true
@univariate false
@dimension {dim}
@equallength true
@serieslength 24
@targetlabel true
@data    
"""

    if not os.path.exists(path):
        os.makedirs(path)

    dim = len(train_instances[0])
    with open(path+file+'_TRAIN.ts','w') as ts_file:
        ts_file.write(header.format(file=file, dim=dim))
        for instance in train_instances:
            line = ":".join(map(str, instance)) + "\n"
            line = line.replace("'","")
            line = line.replace(" ", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace("$", " ")
            ts_file.write(line)

    with open(path+file+'_TEST.ts','w') as ts_file:
        ts_file.write(header.format(file=file, dim=dim))
        for instance in test_instances:
            line = ":".join(map(str, instance)) + "\n"
            line = line.replace("'", "")
            line = line.replace(" ", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            line = line.replace("$", " ")
            ts_file.write(line)


def download_file(path, city, _id):

    fullpath = f'{path}/{city}_{_id}'

    url = f'https://bulk.meteostat.net/v2/hourly/{_id}.csv.gz'
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)
    filename = f'{fullpath}/{_id}.csv.gz'

    # Download the file
    urllib.request.urlretrieve(url, filename)

    # Extract the csv file from the gz file
    with gzip.open(filename, 'rb') as f_in:
        with open(f'{fullpath}/{_id}.csv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def generate_datasets(path, city, _id, target_name=""):
    fullpath = f'{path}/{city}_{_id}/'
    csv_file = f'{fullpath}/{_id}.csv'
    ts_file = f'/{city}_{_id}_{target_name}'

    X_train, X_test, y_train, y_test = read_dataset(csv_file, target_name)
    X_train, X_test = clean_data(X_train, X_test)
    columns = ['dwpt', 'rhum', 'wdir', 'wspd', 'pres', 'temp']
    train_instances, test_instances = format_file(X_train, y_train, X_test, y_test, columns, target_name)
    write_ts_file(fullpath+"/"+target_name, ts_file, train_instances, test_instances)


def generate_datasets_v2(path, city, _id, target_name=""):
    fullpath = f'{path}/{city}_{_id}/'
    ts_file = f'/{city}_{_id}_{target_name}'

    columns = ['pres_atm', 'max_pres_atm', 'min_pres_atm',
               "rad_glob",
               "temp", "dew_point", "max_temp", "min_temp",
               "max_dew", "min_dew",
               "max_humi", "min_humi", "humi",
               "wind_direc", "max_wind", "wind_speed"]

    X_train, X_test, y_train, y_test = read_dataset_v2(fullpath, target_name)
    X_train, X_test = clean_data(X_train, X_test)
    train_instances, test_instances = format_file(X_train, y_train, X_test, y_test, columns, target_name)
    write_ts_file(fullpath+"/"+target_name, ts_file, train_instances, test_instances)


def create_v1():
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
    path = "./datasets/files"
    for key, value in cities.items():
        # download_file(path, key, value)
        generate_datasets(path, key, value, target_name="prcp")


def create_v2():
    cities = {
        "manaus": "a101",
        "teresina": "a312",
        "rio_janeiro": "a652",
        "brasilia": "a001",
        "curitiba": "a807",
    }
    path = r"./datasets/files_v2"
    for key, value in cities.items():
        generate_datasets_v2(path, key, value, target_name="prep_tot")


if __name__ == "__main__":

    create_v2()

