
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Dataset loaded from file
raw_data = '''
8.56   -1.22   4.67   -0.00   -0.15   -0.10   37.79   4.15   1
8.57   -1.24   5.00   0.02   -0.08   -0.04   37.79   4.15   1
8.77   -1.13   4.76   0.02   -0.12   -0.03   37.87   4.15   1
8.71   -0.98   4.86   0.06   -0.14   -0.03   37.79   4.15   1
8.66   -1.00   5.11   0.08   -0.05   -0.06   37.85   4.14   1
8.79   -1.34   4.83   0.05   -0.08   -0.10   37.87   4.14   1
8.09   -1.00   5.13   0.14   -0.04   -0.10   37.79   4.13   1
8.53   -1.23   4.72   0.01   0.11   -0.16   37.83   4.15   1
7.84   -3.50   5.12   -0.04   -0.02   -0.27   37.93   4.12   2
8.41   -2.76   4.53   0.22   0.25   -0.31   37.95   4.11   2
7.81   -3.39   6.08   0.16   -0.30   -0.01   37.95   4.11   2
6.93   -3.36   4.38   -0.02   0.33   0.36   37.99   4.13   2
8.43   -3.18   4.58   -0.23   -0.02   0.03   37.93   4.12   2
7.91   -3.67   5.27   -0.07   -0.70   -0.10   37.93   4.13   2
7.89   -2.84   5.16   0.10   -0.09   -0.27   37.95   4.11   2
8.98   -3.66   4.36   -0.08   0.14   -0.30   38.01   4.13   2
7.87   -2.94   4.66   0.26   -0.11   -0.11   38.01   4.12   2
8.20   -4.08   4.52   0.08   -0.14   -0.19   38.03   4.11   2
8.21   -3.21   5.36   -0.05   0.11   0.06   37.99   4.13   2
8.24   -3.34   4.37   -0.00   0.49   0.29   38.01   4.13   2
7.43   -3.15   5.38   -0.13   -0.18   0.06   37.95   4.13   2
7.34   -3.29   5.75   0.23   -0.17   -0.15   37.93   4.13   2
7.93   -3.72   5.89   -0.03   -0.10   -0.43   37.93   4.12   2
8.75   -3.86   4.75   -0.00   -0.12   -0.08   37.95   4.11   2
8.62   0.15   4.81   0.03   -0.15   -0.14   37.19   4.15   3
8.57   -0.02   5.03   0.14   -0.12   -0.08   37.27   4.14   3
8.35   0.15   5.12   0.10   -0.06   0.02   37.25   4.14   3
8.34   0.11   5.18   0.04   -0.11   -0.02   37.19   4.15   3
8.65   -0.04   5.02   0.01   -0.13   -0.03   37.21   4.15   3
8.47   -0.16   5.18   -0.02   -0.08   -0.02   37.19   4.14   3
8.80   -0.05   5.15   0.09   -0.03   -0.10   37.21   4.15   3
8.57   -0.34   5.27   -0.05   -0.30   -0.09   37.21   4.13   3
8.59   0.05   5.16   0.07   -0.02   -0.07   37.21   4.15   3
8.57   0.06   4.91   -0.07   0.02   -0.13   37.25   4.16   3
'''

# Parse the raw data into a numpy array
data = np.array([list(map(float, line.split())) for line in raw_data.strip().split('\n')])

# Split features and labels
X = data[:, :-1]  # Features (all columns except last)
y = data[:, -1]   # Labels (last column)

# Split into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Probabilistic Neural Network implementation
class ProbabilisticNeuralNetwork:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def gaussian_kernel(self, x, y):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * self.sigma ** 2))

    def predict(self, X):
        predictions = []
        for x in X:
            probabilities = []
            for cls in np.unique(self.y_train):
                cls_indices = np.where(self.y_train == cls)[0]
                cls_samples = self.X_train[cls_indices]
                probabilities.append(sum(self.gaussian_kernel(x, s) for s in cls_samples))
            predictions.append(np.argmax(probabilities) + 1)
        return np.array(predictions)

# Create and train the model with different sigma values
sigma_values = np.linspace(0.1, 3.0, 30)  # Range of sigma values from 0.1 to 3.0
errors = []

# Evaluate the error for different sigma values
for sigma in sigma_values:
    pnn = ProbabilisticNeuralNetwork(sigma=sigma)
    pnn.fit(X_train, y_train)
    
    # Test accuracy
    test_pred = pnn.predict(X_test)
    accuracy = accuracy_score(y_test, test_pred)
    error = 1 - accuracy
    errors.append(error)

print(error)
# Find the best sigma
best_sigma = sigma_values[np.argmin(errors)]
print(f"The best sigma value is {best_sigma:.2f} with the minimum error.")

# Plotting error for different sigma values
plt.figure(figsize=(10, 6))
plt.plot(sigma_values, errors, label='Test Error', color='blue')
plt.title('Error vs. Sigma for PNN')
plt.xlabel('Sigma')
plt.ylabel('Error (1 - Accuracy)')
plt.grid(True)
plt.legend()
plt.show()
