import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Step 1: Create a dataset
'''data={
    'X1': [2.5,0.5,2.2,1.9,3.1,2.3,2.0,1.0,1.5,1.1],
    'X2': [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]
}'''

data= np.loadtxt(r"C:\Users\shirs\OneDrive\Desktop\Training\data.txt")
#data = np.genfromtxt('')


df=pd.DataFrame(data)

#Step 2: Standardize the data
mean = df.mean()
std = df.std()
df_standardized = (df-mean)/std

print("Standardized data:")
print(df_standardized)

#Step 3: Compute the covariance matrix
cov_matrix = np.cov(df_standardized.T)
print("\n Covariance matrix:")
print(cov_matrix)

#Step 4: Compute the eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\n Eigenvalues:")
print(eigenvalues)
print("\n Eigenvectors:")
print(eigenvectors)

#Step 5: Sort eigenvectors and eigenvalues
sorted_idx = eigenvalues.argsort()[::-1]
sorted_eigenvalues = eigenvalues[sorted_idx]
sorted_eigenvectors = eigenvectors[:,sorted_idx]

print("\n Sorted Eigenvalues:")
print(sorted_eigenvalues)
print("\n Sorted Eigenvectors:")
print(sorted_eigenvectors)

#Step 6: Project the data onto principal components
W = sorted_eigenvectors[:,:2] # Select the first principal component
projected_data = df_standardized.dot(W)
print("\n Projected data on PC1:")
print(projected_data)

#optional: Visulaize the data
plot_data = np.array(projected_data)
#Define colors and symbols for each row
colors = ['red','green','blue','purple','orange','cyan']
symbols = ['o','s','^','D','P','*'] #cIRCLE ,SQUARE, TRIANGLE, DIAMOND, PENTAGON, STAR

#Ploting
plt.figure(figsize=(8,6))
group1=16
group2=16

data_modified=np.array(df)
#Select coloums for scatter plot
x=data_modified[:,0] #First column as x-axis
y=data_modified[:,1] #Second column as y-axis
plt.scatter (x,y,color=colors[0],marker=symbols[0],label=f'Raw Data',s=100)
#Labels and legend
plt.xlabel("Coloumn 1")
plt.ylabel("Coloumn 2")
plt.title("raw Data Plot")
plt.legend()
plt.grid()
plt.show()

for i in range(group1):
    x,y = plot_data[i]
    plt.scatter(x,y,color=colors[1],marker=symbols[0],label=f'PC{1}' if i==0 else None ,s=100)


for i in range(group1,group1+group2):
    x,y = plot_data[i]
    plt.scatter(x,y,color=colors[2],marker=symbols[1],label=f'PC{2}' if i==group1 else None ,s=100)


#Labels and legend       
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PCA Plot")
plt.legend()
plt.grid()


#Show the plot
plt.show()

