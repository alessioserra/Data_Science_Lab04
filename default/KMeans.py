import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance
import csv

# EXERCISE 3
class KMeans:
    def __init__(self, n_clusters, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit_predict(self, dataset):
        
        k = 15 # number of cluster we want
        centroids = []
        kluster = {}

        m = 0

        for i in range(15):
            centroids.append(dataset[random.randint(0, len(dataset))])
            kluster[str(i)] = []

        # SHOW RANDOM CENTROIDS
        xklusters = [xk for xk, yk in centroids]  # saving all x and y in different vectors
        yklusters = [yk for xk, yk in centroids]
        colors2 = np.random.rand(15)
        plt.scatter(xklusters, yklusters, c=colors2, alpha=0.5)
        plt.show()

        # MAX 100 iterations
        while m < 10:
            # 2 Assignment 
            for element in dataset:
                minimum = 1000000000
                index = 0
                i = -1
                for centroid in centroids:
                    i = i+1
                    dist = distance.euclidean(element, centroid)
                    if dist < minimum:
                        minimum = dist
                        index = i

                kluster[str(index)].append(element)

            # 3 Update 
            oldCentroids = centroids
            centroids.clear() # new list of centroid
            for i in range(15):
                allX = 0
                allY = 0
                for row in kluster[str(i)]:
                    # new mean values for centroid
                    allX = allX + int(row[0])
                    allY = allY + int(row[1])

                meanX = allX/len(kluster[str(i)])
                meanY = allY/len(kluster[str(i)])
                newCentroid = [meanX, meanY]

                if newCentroid in oldCentroids:

                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    cmap = plt.cm.get_cmap("hsv", len(kluster))
                    
                    self.labels = kluster

                    for key in kluster.keys():
                        xx = [xx for xx, yy in kluster[key]]
                        yy = [yy for xx, yy in kluster[key]]

                        ax2.scatter(xx, yy, cmap(kluster[key]))

                    plt.show()
                    return kluster

                # else:
                centroids.append(newCentroid)

            # new iteration
            m = m + 1
            print("Interation #"+str(m))
        
        self.labels = kluster
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        cmap = plt.cm.get_cmap("hsv", k)

        for key in kluster.keys():
            xx = [xx for xx, yy in kluster[key]]
            yy = [yy for xx, yy in kluster[key]]

            ax2.scatter(xx, yy, cmap(kluster[key]))


        # EXERCISE 1.6
        self.centroids = centroids
        xc = [xx for xx, yy in centroids]
        yc = [yy for xx, yy in centroids]
        area = 30
        ax2.scatter(xc, yc, c="black", s=area, marker="*")
        
        plt.show()

        return kluster


    # EXERCISE 2.4
    def dump_to_file(self, filename, klusters, x, y):
        with open(filename, mode="w", newline="") as csvfile:

            # Headers
            fieldnames = ['Id', 'ClusterId']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            counter = 0

            for xW, yW in zip(x, y):
                for key in klusters.keys():
                    for el in klusters[key]:
                        if el[0] == xW and el[1] == yW:
                            writer.writerow({'Id': str(counter), 'ClusterId': str(key)})
                            counter = counter + 1
        print("Computed Finished")
