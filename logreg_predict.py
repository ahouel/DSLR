from tools import parse_tools as pt
from tools import maths_tools as mt
from tools.constants import FEATURE_DROP, HOUSE_COL
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Predict houses of students in a dataset. Taking a dataset and weights (made from logreg_train.py)
# To calculate the accuracy_score, dataset should be already filled with students houses.

# Check if both weights and dataset files are compatible
def checker(features, weights):
    n = len(features) + 1
    for c in weights:
        if len(weights[c]) != n:
            return False
    return True

# Main prediction function, uses the dataframe and thetas as matrix to calculate
# the sigmoid_function. For each row, we keep the maximum value found : this
# is the house fitting the most for this student.
def predict(df, houses, thetas):
    X = np.concatenate((np.ones((df.shape[0], 1)), np.matrix(df)), axis=1)
    X = mt.clear_nan(X)
    Y = {}
    for house in houses:
        theta = np.matrix(thetas[house])
        Y[house] = mt.sigmoid_function(X, theta).T.tolist()[0]
    Y_mat = np.array([Y[h] for h in Y]).transpose()
    Y_pred = []
    for row in Y_mat:
        max = mt.max(row)
        for i, val in enumerate(row):
            if val == max:
                Y_pred.append(houses[i])
                break
    return Y_pred

# df_data : dataframe of the dataset
# df_weights : datafram of weights (stds, means, and thetas)
# Y_true : real houses, only used for accuracy score if noone is nan
# Y_pred : predicted houses from our function
# is_test : set to false by default, becomes true if a nan value is found in Y_true
# accuracy_score is called only when it is set to False
# features : names of features used to predict houses
# houses : names of students houses
#
# We first get ride of unused columns in df_data.
# Then store stds and means from df_weights, and use them to standardize
# data from df_data. df_weights can now be used as a matrix for our thetas.
# We set Y_pred by calling our predicting function.
# Finally calling the accuracy_score if is_test is false, create a dataframe for
# our predictions and put it into a .csv file.

if __name__ == "__main__":
    df_data, df_weights = pt.parsefile(__file__)
    features = pt.getfeatures(df_data)
    Y_true = df_data[HOUSE_COL].tolist()
    is_test = pt.house_missing(Y_true)
    features = [f for f in features if f not in FEATURE_DROP]
    houses = [h for h in df_weights if h not in ['Std', 'Mean']]
    df_data.drop([c for c in df_data if c not in features], axis=1, inplace=True)
    if not checker(features, df_weights):
        print("Weights and dataset files are not compatible : wrong number of features")
        exit(-1)
    stds = df_weights['Std'].tolist()
    means = df_weights['Mean'].tolist()
    df_weights.drop('Std', axis=1, inplace=True)
    df_weights.drop('Mean', axis=1, inplace=True)
    df_data = mt.standardize(df_data, mean=means, std=stds)
    Y_pred = predict(df_data, houses, df_weights)
    if not is_test:
        print("Accuracy_score :", accuracy_score(Y_true, Y_pred))
    df = pd.DataFrame(Y_pred, columns=[HOUSE_COL])
    print("Generating houses.csv")
    df.to_csv("houses.csv", header=True)