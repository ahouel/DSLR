import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure showing plots of costs functions for each iteration of gradient descent
def cost_plot(costs):
    fig = plt.figure(figsize=(6, 6))
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    for house in costs:
        plt.plot(costs[house], label=house)
    plt.legend()
    fig.suptitle("Cost functions for each house")
    fig.tight_layout()
    plt.show()


# Create a pair plot, showing grades of students from all houses for all different
# features, may takes a few time to load
def pairplot(data):
    print("Generating pair plot, this might takes a few time...")
    sns.pairplot(data, hue="Hogwarts House", markers = ".", height=1, dropna=True)
    plt.tight_layout()
    plt.show()

# Create a scatter plot, showing grades of students from all houses for two
# different features
def scatter_plot(houses, feature_1, feature_2):
    fig = plt.figure(figsize=(6, 6))
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    for house in houses:
        plt.scatter(houses[house][feature_1], houses[house][feature_2], label=house, marker='*')
    fig.tight_layout()
    plt.legend()
    plt.show()

# Plot an histogram for each feature, showing the distribution of students grades for each house
# 1 figure created for 4 features
def histogram(houses, features):
    n = len(features)
    i = 0
    f = 0
    while i < n:
        f += 1
        fig = plt.figure(f, figsize=(16, 6))
        plt.style.use('ggplot')
        for j in range(4):
            if i < n:
                plt.subplot(1, 5, j + 2)
                plt.xlabel('Grades')
                plt.ylabel('Frequency of students')
                for h in houses:
                    try:
                        plt.hist(houses[h][features[i]][0],alpha=0.4, label=h)
                    except:
                        pass
                plt.title(features[i], fontsize=12)
            i += 1
        fig.legend([h for h in houses],loc=2,fontsize=16,shadow=True)
        fig.suptitle('Distribution of Students Grades for Each House', fontsize=18)
        fig.tight_layout()
    plt.show()