import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Step 1: Create the dataset
data=np.loadtxt(r'C:\Users\shirs\OneDrive\Desktop\Training\data.txt')
#data =np.genfromtxt("C:\Users\shirs\OneDrive\Desktop\Training\data.txt", )

df=pd.DataFrame(data)
#Number of clusters
k = 2


#Step 2: Initialize the centroids randomly
np.random.seed(42) #for reproducibility
initial_indices = np.random.choice(len(data), k, replace=False)
centroids = data[initial_indices]

#Function to calculate the Distance
def calculate_distance(point, centroid):
    return np.linalg.norm(point - centroid, axis=1)

#Step 3: Iterate to assign cluster and update centroids
max_iterations = 10
for iterations in range (max_iterations):
    #Assign points the neaarest centroid
    cluster_labels = np.array([
        np.argmin(calculate_distance(point, centroids))
        for point in data
    ])
    #Calculate new centroids
    new_centroids = np.array([
        data[cluster_labels == i].mean(axis=0)
        for i in range(k)
    ])
    #Check for convergence
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

#Step 4: Visualize the results
for i in range (k):
    cluster_points = data[cluster_labels == i]
    plt.scatter(cluster_points[:,0], cluster_points[:,1], label=f'Cluster {i+1}')
plt.scatter(centroids[:,0], centroids[:,1], color='black', marker='x', label='Centroids')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.show()