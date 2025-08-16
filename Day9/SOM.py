import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Input dataset

data = np.array([
  [1,0,1,0],
  [1,0,0,0],
  [1,1,1,1],
  [0,1,1,0]
])

#df= np.loadtxt('')
#data = pd.DataFrame(df)
#SOM parameters
n_clusters = 2 
iterations = 10
alpha_initial = 0.5
sigma_initial = 1.0

# Initialize weights randomly
np.random.seed(42)
weights= np.random.rand(n_clusters,data.shape[1])

# Fuction to decay learning rate and neighboure radius
def decay(parameter, initial_value, iteration, total_iterations):
  return initial_value * (1-iteration/total_iterations)

# Training the SOM
for t in range(iterations):
    alpha = decay(alpha_initial, alpha_initial, t, iterations)
    sigma = decay(sigma_initial, sigma_initial, t, iterations)
    
    for x in data:
        distance = np.linalg.norm(weights-x,axis=1)
        bmu_idx=np.argmin(distance)

        for i in range(n_clusters):
           distance_to_bmu = np.abs(i - bmu_idx)
           if distance_to_bmu <= sigma:
              influence=np.exp(-distance_to_bmu*2 / (2*sigma*2))
              weights[i] +=alpha*influence*(x-weights[i])

cluster_assignments = [np.argmin(np.linalg.norm(weights - x,axis=1)) for x in data]

def plot_clusers(data,cluster_assignments,weights):
  colors = ['r','g','b','y']
  plt.figure(figsize=(8,6))

  '''
   # Plot raw data points 
   for i,x in enumerate(data):
      plt.scatter(x[0],x[2], color=colors[cluster_assignments[i]], label=f"Data {i+1}")
  '''

  # Plot clusters cluster
  for i,w in enumerate(weights):
     plt.scatter(w[0],w[2],color=colors[i],edgecolors='k',s=200,label=f"Cluster {i+1} Center")
  
  plt.title("SOM clustering Results")
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 3")
  plt.legend()
  plt.grid()
  plt.show()

plot_clusers(data,cluster_assignments,weights)
print("The cluster of crossponding to test data are ")
for i in range(len(cluster_assignments)):
   print(cluster_assignments[i])