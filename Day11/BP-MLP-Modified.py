import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Raw data
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

# One-hot encoding the labels
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y.astype(int) - 1]

y_encoded = one_hot_encode(y, 3)

# Split into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLP with Backpropagation
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Weights initialization
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        # Backward pass (Gradient computation)
        m = X.shape[0]

        # Output layer error
        self.output_error = self.a2 - y
        self.output_delta = self.output_error * self.sigmoid_derivative(self.a2)

        # Hidden layer error
        self.hidden_error = self.output_delta.dot(self.W2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.a1)

        # Gradients
        self.W1_grad = X.T.dot(self.hidden_delta)
        self.b1_grad = np.sum(self.hidden_delta, axis=0)
        self.W2_grad = self.a1.T.dot(self.output_delta)
        self.b2_grad = np.sum(self.output_delta, axis=0)

    def update_weights(self):
        # Update weights and biases
        self.W1 -= self.learning_rate * self.W1_grad
        self.b1 -= self.learning_rate * self.b1_grad
        self.W2 -= self.learning_rate * self.W2_grad
        self.b2 -= self.learning_rate * self.b2_grad

    def train(self, X, y, epochs=1000):
        errors = []
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            self.update_weights()
            # Compute error
            loss = np.mean(np.square(self.a2 - y))
            errors.append(loss)
        return errors

    def test(self, X, y):
        # Forward pass on test data
        predictions = self.forward(X)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred == y_true)
        return accuracy

    def save_model(self, filename):
        # Save model weights and biases using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        # Load model from file
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

# Create and train the MLP model
mlp = MLP(input_size=X_train.shape[1], hidden_size=5, output_size=3, learning_rate=0.01)
errors = mlp.train(X_train, y_train, epochs=2000)

# Save the trained model
mlp.save_model("mlp_model.pkl")

# Plot training error
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Training Error over Epochs')
plt.show()

# Test the trained model
loaded_mlp = MLP.load_model("mlp_model.pkl")
accuracy = loaded_mlp.test(X_test, y_test)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Example: Test on new data
t_data='''8.85   -1.13   4.70   0.11   -0.18   -0.05   37.85   4.13
8.60   -1.03   5.06   0.14   0.23   -0.06   37.83   4.13
8.92   -0.96   4.91   0.06   0.03   -0.08   37.79   4.15
8.44   -0.24   5.12   -0.01   -0.20   -0.09   37.19   4.15
8.61   -0.33   5.18   0.05   -0.07   -0.01   37.25   4.16
8.80   -0.44   5.24   0.04   -0.06   -0.05   37.13   4.14
8.64   -0.14   5.06   -0.01   -0.10   -0.08   37.21   4.15
'''
#new_data = np.array([[8.65, -0.04, 5.02, 0.01, -0.13, -0.03, 37.21, 4.15]])
# Parse the raw data into a numpy array
new_data = np.array([list(map(float, line.split())) for line in t_data.strip().split('\n')])

# Normalize the new data using the same scaler
new_data_scaled = scaler.transform(new_data)

# Get the output from the model (probabilities for each class)
prediction_probabilities = loaded_mlp.forward(new_data_scaled)

# Get the predicted class (index of the maximum value)
for data in prediction_probabilities:
    predicted_class = np.argmax(data)
    print(f"Predicted class for the new data: {predicted_class + 1}")