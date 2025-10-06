#Implement Hebb’s Rule for Basic Logical Functions.

import numpy as np

# Convert binary values (0,1) to bipolar (-1,1)
def bipolar(x):
    return np.where(x == 0, -1, 1)

# Predict output using current weights
def predict(weights, X):
    s = np.dot(X, weights)
    return np.where(s >= 0, 1, -1)

# Hebbian learning algorithm with verbose updates per epoch
def train_hebb_verbose(X, y, lr=0.1, epochs=5):
    w = np.zeros(X.shape[1])  # Initialize weights to zeros
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}")
        for xi, yi in zip(X, y):
            # Hebb's rule: Δw = η * y * x
            w += lr * yi * xi
            print(f"Input: {xi}, Target: {yi}, Updated Weights: {w}")
        
        # Predictions after each epoch
        preds = predict(w, X)
        preds01 = np.where(preds == -1, 0, 1)
        print(f"Predictions after epoch {epoch}: {preds01.tolist()}")
    
    return w

# Input patterns with bias term (first column = 1)
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# Choose logic gate
choice = input("Enter gate (AND/OR): ").strip().upper()

if choice == "AND":
    targets = np.array([0, 0, 0, 1])
elif choice == "OR":
    targets = np.array([0, 1, 1, 1])
else:
    print("Invalid choice! Please enter AND or OR.")
    exit()

# Convert targets to bipolar form
y = bipolar(targets)

# Train using Hebbian learning
final_w = train_hebb_verbose(X, y, lr=0.2, epochs=5)

print("\nFinal Weights:", final_w)
