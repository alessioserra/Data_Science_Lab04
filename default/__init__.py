import numpy as np
import matplotlib.pyplot as plt
# using class import
from KMeans import KMeans

# EXERCISE 1
# read data set with numpy
data = np.loadtxt("gauss_clusters.txt", delimiter=",", skiprows=1)

x = [x for x, y in data] # saving all x and y in different vectors
y = [y for x, y in data]

# scanner plot
colors = np.random.rand(len(data))
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()


# GO function
klusters = KMeans(15, 100)
k = klusters.fit_predict(data)
klusters.dump_to_file("result.csv", k, x, y)

"""
# EXERCISE 2
data2 = np.loadtxt("camaleon_clusters.txt", delimiter=",", skiprows=1)

x2 = [x for x, y in data2] # saving all x and y in different vectors
y2 = [y for x, y in data2]

# scanner plot
colors = np.random.rand(len(data2))
plt.scatter(x2, y2, c=colors, alpha=0.5)
plt.show()

# GO function
klusters = KMeans(15, 100)
k2 = klusters.fit_predict(data2)
klusters.dump_to_file("result.csv", k2, x2, y2)
"""