from tools import parse_tools as pt
from tools import maths_tools as mt
from tools import visualization_tools as vt

# Compare features two by two to see the most similar ones, using Pearson's correlation
# coefficient. Stores it in <corr> indexs 0 for the absolute value, 1 for the value,
# 2 for the first feature and 3 for the second one.
# Then show a scatter plot of students grades for those features.
# houses { <name> : [<feature> : notes]}

if __name__ == "__main__":
    df = pt.parsefile(__file__)
    features = pt.getfeatures(df)
    n = len(features)
    if n < 2:
        print("At least two features are required, exiting...")
        exit(-1)
    houses = pt.gethouses(df)
    corr = [0, 0, "", ""]
    for i in range(n):
        for j in range(i, n):
            tmp_corr = mt.corr(df[features[i]], df[features[j]])
            abs_corr = mt.abs(tmp_corr)
            if  abs_corr > corr[0]:
                corr[0] = abs_corr
                corr[1] = tmp_corr
                corr[2] = features[i]
                corr[3] = features[j]
    for house in houses:
        dfh = df[df["Hogwarts House"] == house]
        houses[house][corr[2]] = dfh[corr[2]]
        houses[house][corr[3]] = dfh[corr[3]]

    print("The most similar features are :", corr[2], "and", corr[3],
    "\nPearson's correlation coefficient is close to", corr[1])
    vt.scatter_plot(houses, corr[2], corr[3])