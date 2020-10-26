import os.path
import numpy as np
import pandas as pd
import argparse
from tools.constants import PREDICT_SCRIPT, DATE_FORMAT
from datetime import datetime

# Check if a value is missing in the list. If not, returns false.
def house_missing(lst):
    for val in lst:
        if (isinstance(val, str) and val == "") or (isinstance(val, float) and np.isnan(val)):
            return True
    return False


# Allows only .csv format to be read by the parser
def only_csv(parser, fname):
    ext = os.path.splitext(fname)[1][1:]
    if ext != "csv":
       parser.error("file doesn't end with .csv")
    return fname

# Generic parsing for python scripts. Open a file.csv and returns a dataframe of it,
# doing the same for weights in case the script calling is the predicting one.
def parsefile(file):
    fname = os.path.basename(file)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='the dataset file (only .csv)', type=lambda s:only_csv(parser, s))
    if fname == PREDICT_SCRIPT:
        parser.add_argument('weights', help='the thetas (only .csv)', type=lambda s:only_csv(parser, s))
    args = parser.parse_args()
    try:
        df_data = pd.read_csv(args.dataset, index_col=0)
        if fname == PREDICT_SCRIPT:
            df_weights = pd.read_csv(args.weights, index_col=0)
            return df_data, df_weights
    except FileNotFoundError as e:
        print(e)
        exit(-1)
    return df_data

# Parse features from a dataframe, returning a list of feature name with their type
# features [name, type] if set_type is True, else features [name]
def getfeatures(df, set_type=False):
    features = []
    for column, feature, dtype in zip(df, df.columns, df.dtypes):
        s = str(dtype)
        if s.startswith("float") or s.startswith("int"):
            if set_type:
                features.append([feature, s])
            else:
                features.append(feature)
        elif isdateformat(df[column]) and set_type:
            features.append([feature, "date"])
    return features

# Parse different type of Hogwarts Houses in the datafram and store each unique case in a dict
# then returns it
def gethouses(df):
    houses = {}
    try:
        for house in df["Hogwarts House"]:
            if (isinstance(house, str) and house == "") or (isinstance(house, float) and np.isnan(house)):
                continue
            elif not house in houses:
                houses[house] = {}
    except:
        print("No column \"Hogwarts House\" found, exiting...")
        exit(-1)
    if not len(houses):
        print("No value found in \"Hogwarts House\", exiting...")
        exit(-1)
    return houses

# Verify if the values in <lst> are matching a date format
def isdateformat(lst: list):
    ret = False
    try:
        for date in lst:
            if (isinstance(date, str) and date != "") or (isinstance(date, float) and not np.isnan(date)):
                datetime.strptime(date, DATE_FORMAT)
                ret = True
    except:
        return False
    return ret

# Change values of the <lst> from a string to a float (timestamp)
def totimestamps(lst: list):
    times = []
    for date in lst:
        try:
            times.append(datetime.timestamp(datetime.strptime(date, DATE_FORMAT)))
        except:
            pass
    return times

# Takes a timestamp and returns a string (date format)
def fromtimestamp(time: float):
    return datetime.fromtimestamp(time).strftime(DATE_FORMAT)