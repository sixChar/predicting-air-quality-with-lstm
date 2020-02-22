import pandas as pd
import numpy as np
import os,random
import json

# Columns of the data: 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station'
# wd is wind direction and WSPM is wind speed (m/s)


DATA_FOLDER_PATH = "PATH_TO_DATA" # Replace this.
assert DATA_FOLDER_PATH != "PATH_TO_DATA" # Remove this. It is here to make sure you replace the above path with a correct one.


STATIONS = ['Tiantan', 'Nongzhanguan', 'Huairou', 'Wanshouxigong', 'Dongsi', 'Gucheng', 'Shunyi', 'Wanliu', 'Dingling', 'Aotizhongxin', 'Guanyuan', 'Changping']
ATTRIBUTES = ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','wd','WSPM']

NUM_STATIONS = 12
TRAIN_EXAMPLES = 28000

DATE_COLUMNS = ['year','month','day','hour']

# Columns in the neural net input that are not Nan flags
NON_NAN_FLAG_COLUMNS = 156


# Direction vectors for the cardinal directions
CARDINAL_VECTORS = {'N':np.array([0,1]),
                    'E':np.array([1,0]),
                    'S':np.array([0,-1]),
                    'W':np.array([-1,0])}



# Paths to files for the mean and standard deviation of the training data.
TRAIN_MEANS_PATH = "data_stats/means"
TRAIN_STDS_PATH = "data_stats/stds"

'''
    Method used to calculate and save the means and stds of the numerical columns
'''
def calc_and_save_train_stats():
    data = load_data_raw()
    data = data.set_index(['No'])

    data_no_dates = data[data.columns.difference(DATE_COLUMNS)]

    means, stds = get_train_stats(data_no_dates)

    with open(TRAIN_MEANS_PATH, 'w') as f:
        json.dump(means, f)

    with open(TRAIN_STDS_PATH, 'w') as f:
        json.dump(stds, f)

if not (os.path.exists(TRAIN_MEANS_PATH) and os.path.exists(TRAIN_STDS_PATH)):
    calc_and_save_train_stats()

with open(TRAIN_MEANS_PATH) as f:
    TRAIN_MEANS = json.load(f)

with open(TRAIN_STDS_PATH) as f:
    TRAIN_STDS = json.load(f)





'''
    Checks whether the given string s (should be a property/column in the data) corresponds to a wind direction property
'''
def is_wind_dir(s):
    return s[:2].lower() == 'wd'

def make_prop_index():
    prop_index = {}
    i = 0
    for j,key in enumerate(sorted(TRAIN_MEANS.keys())):
        if is_wind_dir(key):
            # Add 2 since each wind direction accounts for 2 property columns in the model format.
            # Keep both regular and adjusted index to make calculating nan flag position easier
            prop_index[key] = i
            prop_index[key+"_NAN"] = j
            i += 2
        else:
            prop_index[key] = i
            i += 1
    return prop_index
#Dictionary holding the index of each given prop in the model format. Used in prop_to_index.
PROP_INDEX = make_prop_index()



'''
    Load the data into a single dataframe.
'''
def load_data_raw():
    data_fnames = os.listdir(DATA_FOLDER_PATH)
    data_files = []
    for fname in data_fnames:
        data_files.append(pd.read_csv(DATA_FOLDER_PATH + fname))
    dates = data_files[0][['No'] + DATE_COLUMNS]
    non_dates = []
    for df in data_files:
        df_no_date = df[df.columns.difference(['No'] + DATE_COLUMNS)]
        station = df_no_date['station'].unique()[0]
        df_no_date = df_no_date.drop('station',axis=1)
        df_no_date = df_no_date.add_suffix('_%s'%station)
        non_dates.append(df_no_date)
    return pd.concat([dates] + non_dates, axis=1)


'''
    Get a consecutive block of a given array of a given size. If a start is not given it is chosen randomly.
'''
def get_consecutive(data,size,start=None):
    first_index = data.index[0]
    num_indices = data.index.shape
    if start == None:
        start = random.randint(first_index,num_indices-size)
    readings = data[(data.index >= start) & (data.index < start + size)]
    return readings


'''
    Processes data into a form suitable for feeding into the neural net model. Wind speed is changed from categorical to 2d numerical vector, nan flags for every value are
    added which tell the network which values were actually nan, nan values are replaced with 0, and numerical properties are normalized (nans do not affect this
    normalization). Note: There will be fewer nan flags than processed data columns since the wind direction is split into 2 properties.
'''
def to_net_input(data):
    #Flags that tell the network whether a given value was nan(i.e. no/bad data). Appended at the end as opposed to right after the values they point to because a feedforward
    #layer will be first so the position doesn't matter.
    nan_flags = np.zeros(data.shape)
    nan_flags[data.isna()] = 1

    # Create array that will be the input to the network (add an extra column for each stations wind direction)
    data_vals = np.zeros([data.shape[0],data.shape[1] + NUM_STATIONS])


    # Change the wind direction columns to numerical vectors and copy values over.
    # Columns are sorted so that the order of the input columns is definitely alphabetical to make decoding network output easier.
    # Note: The "sorted(...)" may be redundant but I couldn't find anything online describing whether or not iterating over a
    #       dataframe's columns is always done alphabetically.
    i = 0
    for col in sorted(data.columns):
        if is_wind_dir(col):
            for j,wind_dir in enumerate(data[col]):
                # If wind_dir isn't a string then it is nan
                if not isinstance(wind_dir,str):
                    wind_vec = np.zeros(2)
                else:
                    wind_vec = wind_dir_to_vec(wind_dir)

                data_vals[j,i] = wind_vec[0]
                data_vals[j,i+1] = wind_vec[1]
            i += 2
        else:
            data_vals[:,i] = preprocess_col(data[col],col)
            i += 1

    return np.concatenate([data_vals,nan_flags],axis=1)


'''
    Replaces nan with zeros and normilizes a 1d array (a column from data)
'''
def preprocess_col(column,column_name,use_train_only=True):
    if use_train_only:
        mean = TRAIN_MEANS[column_name]
        std = TRAIN_STDS[column_name]
    else:
        mean = np.mean(column)
        std = np.std(column)
    return np.nan_to_num((column - mean) / std)


'''
    De-normalizes the numerical predictions from the model (excludes wind direction and nan flags)
'''
def deprocess(model_out):
    result = np.zeros(model_out.shape)
    for i,key in enumerate(sorted(TRAIN_MEANS.keys())):
        if TRAIN_MEANS is None:
            result[...,i] = model_out[...,i]
        else:
            result[...,i] = model_out[...,i] * TRAIN_STDS[key] + TRAIN_MEANS[key]
    return result


'''
    Returns the mean and standard deviations of the training data.
'''
def get_train_stats(data):
    means = {}
    stds = {}
    for col in data.columns:
        # If column is not wind direction
        if not is_wind_dir(col):
            means[col] = np.mean(data[col][:TRAIN_EXAMPLES])
            stds[col] = np.std(data[col][:TRAIN_EXAMPLES])
        else:
            means[col] = None
            stds[col] = None
    return means,stds




'''
    Changes a wind direction string representing a direction (e.g. NNW) into a 2d vector where the first d corresponds to north (+) vs south (-)
    and the second corresponds to west (-) and east (+).
'''
def wind_dir_to_vec(wind_dir):
    # Set wind_dir to upper case, just in case
    wind_dir = wind_dir.upper()
    if len(wind_dir) < 1:
        raise(ValueError("Empty string given to wind_dir_to_vec. Input must be a cardinal or intercardinal direction"))
    if len(wind_dir) == 1:
        return CARDINAL_VECTORS[wind_dir]
    else:
        vec_sum = (CARDINAL_VECTORS[wind_dir[0]] + wind_dir_to_vec(wind_dir[1:]))
        return vec_sum / np.linalg.norm(vec_sum)


def load_all_preprocessed():
    data = load_data_raw()
    data = data.set_index(['No'])

    data_dates = data[DATE_COLUMNS]
    data_no_dates = data[data.columns.difference(DATE_COLUMNS)]

    data_vals = to_net_input(data_no_dates)
    return data_vals, data_dates



'''
    Calculates the index of a given property in the input/output data format that the model uses. For wind direction, only
    the first of 2 indices is given. When nan_flag is true, returns the index of the nan flag corresponding to the given
    property instead.
'''
def prop_to_index(prop,nan_flag=False):
    if nan_flag:
        if is_wind_dir(prop):
            return PROP_INDEX[prop + "_NAN"] + NON_NAN_FLAG_COLUMNS
        else:
            return PROP_INDEX[prop] + NON_NAN_FLAG_COLUMNS
    else:
        return PROP_INDEX[prop]














