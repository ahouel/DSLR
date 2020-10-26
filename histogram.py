from tools import maths_tools as mt
from tools import parse_tools as pt
from tools import visualization_tools as vt
import numpy as np

# Prepares the data to be shown on a histogram plot
# houses { name : features [feature : [0 : [grades], [1 : stats {stat : value}]]]}
# Costs are calculated by first standardizing grades, and then adding std of each same 
# stat between houses, the lower one is the most homogeneous feature.
# costs { feature : value }

if __name__ == "__main__":
    df = pt.parsefile(__file__)
    costs = {}
    houses = pt.gethouses(df)
    features = pt.getfeatures(df)   
    for house in houses:
        dfh = df[df["Hogwarts House"] == house]
        for feature in features:
            grades = dfh[feature]
            stats = {}
            column = [c for c in grades if not np.isnan(c)]
            column = sorted(column)
            n = len(column)
            if n == 0:
                column.append(np.float("NaN"))
            stats["Min"] = column[0]
            stats["25%"] = column[int(n * 0.25)]
            stats["50%"] = column[int(n * 0.50)]
            stats["75%"] = column[int(n * 0.75)]
            stats["Max"] = column[-1]
            houses[house][feature] = grades, stats
    for feature in features:
        costs[feature] = 0
        for stat in ["Min", "25%", "50%", "75%", "Max"]:
            values = []
            for house in houses:
                values.append(houses[house][feature][1][stat])
            values = [(value - mt.mean(df[feature])) / mt.std(df[feature]) for value in values]
            costs[feature] += mt.std(values)

    print("The course with the most homogeneous grades between Hogwarts Houses is", mt.min_dict(costs))
    vt.histogram(houses, features)