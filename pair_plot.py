from tools import parse_tools as pt
from tools import visualization_tools as vt

# Create a pair plot, showing grades of students from all houses for all different
# features, may takes a few time to load

if __name__ == "__main__":
    df = pt.parsefile(__file__)
    features = pt.getfeatures(df) + ["Hogwarts House"]
    vt.pairplot(df[features])