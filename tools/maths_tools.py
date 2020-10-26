import numpy as np

# Sigmoid function
def sigmoid_function(X, theta):
    return 1.0 / (1 + np.exp(-1 * (X * theta.T)))

# Returns the median value of a list
def median(lst):
    s_lst = list(filter(lambda x: not np.isnan(x), sorted(lst)))
    return s_lst[int(len(s_lst) / 2)]

# Standardize data
def standardize(data, mean=[], std=[]):
    if len(mean) == 0 and len(std) == 0:
        return (data - data.mean()) / data.std()
    else:
        for i, c in enumerate(data):
            data[c] = (data[c] - mean[i]) / std[i]
        return data

# Returns the minimum value of a list
def min(lst: list):
    lst = filter(lambda x: not np.isnan(x), lst)
    mn = np.float("NaN")
    for val in lst:
        if np.isnan(mn) or mn > val:
            mn = val
    return mn

# Returns the maximum value of a list
def max(lst: list):
    lst = filter(lambda x: not np.isnan(x), lst)
    mx = np.float("NaN")
    for val in lst:
        if np.isnan(mx) or mx < val:
            mx = val
    return mx

# Returns the sum of the values in the <lst>
def sum(lst: list):
    lst = filter(lambda x: not np.isnan(x), lst)
    sum = 0
    for v in lst:
        sum += v
    return sum

# Returns the absolute value of <value>
def abs(value: float):
    if value > 0:
        return value
    return -value

# Returns the mean of the values in the <lst>
def mean(lst: list):
    i = 0
    mn = 0
    for e in lst:
        if not np.isnan(e):
            mn += e
            i += 1
    if not i:
        return np.float("NaN")
    return mn / i

# Returns the number of elements of the <value> in <lst>
# If <value> is zero, count the total on element float in <lst>
def count(lst: list, value: float = 0):
    count: int = 0
    for e in lst:
        if value:
            if e == value:
                count += 1
        elif not value:
            if not np.isnan(e):
                count += 1
    return count

# Returns the variance of the <lst>
def var(lst: list):
    mn = mean(lst)
    return mean([(v - mn) ** 2 for v in lst])

# Returns the standard derivation of the <lst>
def std(lst: list):
    return var(lst) ** 0.5

# Returns the key of the minimum value in a dict
def min_dict(dct :dict):
    min = np.float("NaN")
    for k in dct:
        if np.isnan(min) or dct[k] < min:
            min = dct[k]
            key = k
    return key

# Get ride of every Nan value in a matrix and replace it with the median of it's column
def clear_nan(X):
    mat = np.array(X.T)
    for i, column in enumerate(mat):
        med = median(column)
        for j, value in enumerate(column):
            if np.isnan(value):
                mat[i, j] = med
    return np.matrix(mat.T)

# Find Pearson's correlation coef between two lists
def corr(lst1, lst2):
    n = len(lst1)
    if len(lst2) != n:
        return np.float("NaN")
    x = 0
    y = 0
    x_s = 0
    y_s = 0
    xy = 0
    for v1, v2 in zip(lst1, lst2):
        if np.isnan(v1) or np.isnan(v2):
            n -= 1
            continue
        x += v1
        y += v2
        xy += v1 * v2
        x_s += v1 ** 2
        y_s += v2 ** 2
    return ((n * xy) - x * y) / ((n * x_s - x ** 2) * (n * y_s - y ** 2)) ** 0.5