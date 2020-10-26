from tools import parse_tools as pt
from tools import maths_tools as mt
from tools import visualization_tools as vt
from tools.constants import CYCLES, ALPHA, OUTPUT, HOUSE_COL, FEATURE_DROP
import pandas as pd
import numpy as np

# Training function which tries to create thetas depending on grades students have for each feature
# and their house. Uses logistic regression (with gradient descent algorithm) as a classifier.
# One-vs-all method, means for each house we consider rest as zero.

# Gradient descent algorithm : try to find the lower value of the cost function
# Thetas are initialized at zero.
def gradient_descent(X, Y):
    theta = np.matrix([np.zeros(X.shape[1])])
    costs = []
    for _ in range(CYCLES):   
        sig = mt.sigmoid_function(X, theta)
        theta = theta - ((ALPHA / X.shape[0]) * ((sig - Y.T).T * X))
        cost = - (Y * np.log(sig)) - (1 - Y) * np.log(1 - sig)
        cost /= Y.shape[1]
        costs.append(cost.item())        
    return theta.tolist()[0], costs

# Logistic regression, one-vs-all classifier
# Y is a matrix of ones (for the current house) and zeros (for the others) : one-vs-all method
# X is a matrix of our data standardized, nan value are also cleaned, and a row of ones is 
# added for the first thetas.
# Then call the gradient descent function with these matrix.
def logistic_regression(df, house):
    Y = np.matrix([1 if h == house else 0 for h in df[HOUSE_COL]])
    df.drop(HOUSE_COL, axis=1, inplace=True)
    X = np.concatenate((np.ones((df.shape[0], 1)), np.matrix(mt.standardize(df))), axis=1)
    X = mt.clear_nan(X)
    return gradient_descent(X, Y)

# df : dataframe of the dataset
# features : names of features used to predict houses
# houses : names of students houses
# costs : dictionnary keeping values of the cost function for each house
# thetas : dictionnary keeping values of thetas for each house
# means, stds : store these values in dataframes (will be used to standardize data in our
# predicting script)
#
# We call the logistic regression function for each house.
# Then joining thetas calculated to means and stds in a dataframe and put it into
# a .csv file which will be read by our prediction script.
# Also show a plot with the cost functions for each house.

if __name__ == "__main__":
    df = pt.parsefile(__file__)
    features = pt.getfeatures(df)
    houses = pt.gethouses(df)
    costs = {}
    thetas = {}
    features = [f for f in features if f not in FEATURE_DROP]
    df.drop([c for c in df if c not in features + [HOUSE_COL]], axis=1, inplace=True)
    means = pd.DataFrame(df.mean().tolist(), columns=['Mean'])
    stds = pd.DataFrame(df.std().tolist(), columns=['Std'])
    for house in houses:
        thetas[house], costs[house] = logistic_regression(df.copy(), house)
    output = pd.DataFrame.from_dict(thetas).join(means).join(stds)
    output.to_csv(OUTPUT, header=True)
    vt.cost_plot(costs)