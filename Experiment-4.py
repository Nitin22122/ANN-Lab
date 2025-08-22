#Plot various types of Activation Functions used in Artificial Neural Networks (ANN) and evaluate their performance on the Iris dataset.

import matplotlib.pyplot as plt
import numpy as np

# Generate sample input
x = np.linspace(-10, 10, 1000)

# Activation functions
def step(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def linear(x):
    return x

# Plot settings
plt.figure(figsize=(12, 8))

# Dictionary of functions
functions = {
    'Step': step,
    'Sigmoid': sigmoid,
    'Tanh': tanh,
    'ReLU': relu,
    'Leaky ReLU': leaky_relu,
    'Linear': linear
}

# Plot each activation
for i, (name, func) in enumerate(functions.items()):
    plt.subplot(2, 3, i+1)
    plt.plot(x, func(x))
    plt.title(name)
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')

plt.tight_layout()
plt.suptitle("Common Activation Functions", fontsize=16, y=1.02)
plt.show()
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Activation functions to test
activation_list = ['identity', 'logistic', 'tanh', 'relu']
results = {}

# Train and evaluate for each activation
for activation in activation_list:
    model = MLPClassifier(
        hidden_layer_sizes=(10,),
        activation=activation,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[activation] = acc
    print(f"Activation: {activation}, Accuracy: {acc:.4f}")
