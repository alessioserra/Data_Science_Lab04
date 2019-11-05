import numpy as np
import matplotlib.pyplot as plt
# using class import
"""
from default.KMeans import KMeans
"""
from sklearn.cluster import KMeans

# EXERCISE 1
# read data set with numpy
data = np.loadtxt("gauss_clusters.txt", delimiter=",", skiprows=1)

x = [x for x, y in data] # saving all x and y in different vectors
y = [y for x, y in data]

# scanner plot
colors = np.random.rand(len(data))
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()


"""
# GO function
klusters = KMeans(15, 100)
k = klusters.fit_predict(data)
klusters.dump_to_file("result.csv", k, x, y)


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

"""
# Update: using Sci-Kit
"""
n = 15
n_clusters = n
km = KMeans(n_clusters)
pred = km.fit_predict(data)
klusters = {} # empty dictionary

# create index for klusters
for i in range(n):
    klusters[str(i)] = [] 

# Union data and respective clusters:
for point, k in zip(data, pred):
    klusters[str(k)].append(point)
    

fig1, ax2 = plt.subplots(figsize=(8, 5))
cmap = plt.cm.get_cmap("hsv", n)

for key in klusters.keys():
    xx = [xx for xx, yy in klusters[key]]
    yy = [yy for xx, yy in klusters[key]]

    ax2.scatter(xx, yy, cmap(klusters[key]))

"""
Exercise 1.6
"""
centroids = []

for key in klusters.keys():
    
        xx = [x for x, y in klusters[key]]
        yy = [y for x, y in klusters[key]]
    
        xc = np.average(xx)
        yc = np.average(yy)
        centroid = [xc, yc]
        centroids.append(centroid)
        
            
xc = [xx for xx, yy in centroids]
yc = [yy for xx, yy in centroids]
area = 30
ax2.scatter(xc, yc, c="black", s=area, marker="*")
        
plt.show()



