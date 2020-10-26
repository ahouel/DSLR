from tools import maths_tools as mt
from tools import parse_tools as pt
import pandas as pd
import numpy as np

# Open a csv file and print different statistics of it

def statistics(stats, column):
    column = [c for c in column if not np.isnan(c)]
    column = sorted(column)
    n = len(column)
    if n == 0:
        column.append(np.float("NaN"))
    stats["Count"].append(mt.count(column))
    stats["Mean"].append(mt.mean(column))
    stats["Std"].append(mt.std(column))
    stats["Min"].append(column[0])
    stats["25%"].append(column[int(n * 0.25)])
    stats["50%"].append(column[int(n * 0.50)])
    stats["75%"].append(column[int(n * 0.75)])
    stats["Max"].append(column[-1])

if __name__ == "__main__":
    df = pt.parsefile(__file__)
    features = pt.getfeatures(df, set_type=True)
    stats = {
        "Count": [],
        "Mean": [],
        "Std": [],
        "Min": [],
        "25%": [],
        "50%": [],
        "75%": [],
        "Max": []
    }
    for column in features:
        c = column[0]
        if column[1].startswith("date"):
            statistics(stats, pt.totimestamps(df[c]))
            for c in stats:
                if c != "Count":
                    stats[c][-1] = pt.fromtimestamp(stats[c][-1])        
        else:
            statistics(stats, df[c])
    print(pd.DataFrame(stats, index=[lst[0] for lst in features]).T)