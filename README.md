# -Neural-Network-for-Image-Recognition-of-Letters-A-B-and-C
This project demonstrates the implementation of a feedforward neural network from scratch using only NumPy, without relying on machine learning libraries like TensorFlow or PyTorch. The network is trained to classify synthetic 5×6 binary pixel images of the letters A, B, and C.

Features

Fully connected feedforward neural network with one hidden layer

Sigmoid activation function for hidden and output layers

Custom backpropagation and weight updates

Visualization of training loss and accuracy

Prediction and display of input letter images

Dataset

The dataset is synthetic, generated using binary pixel patterns for letters A, B, and C.

Each image is a 5×6 grid flattened into a 30-element array.

Labels are one-hot encoded.
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create Binary Image Data
# -----------------------------

A = np.array([
    0,1,1,1,0,
    1,0,0,0,1,
    1,1,1,1,1,
    1,0,0,0,1,
    1,0,0,0,1,
    1,0,0,0,1
])

B = np.array([
    1,1,1,1,0,
    1,0,0,0,1,
    1,1,1,1,0,
    1,0,0,0,1,
    1,0,0,0,1,
    1,1,1,1,0
])

C = np.array([
    0,1,1,1,1,
    1,0,0,0,0,
    1,0,0,0,0,
    1,0,0,0,0,
    1,0,0,0,0,
    0,1,1,1,1
])

X = np.array([A, B, C])

# One-hot encoded labels
Y = np.array([
    [1,0,0],  # A
    [0,1,0],  # B
    [0,0,1]   # C
])

# -----------------------------
# 2. Activation Functions
# -----------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# -----------------------------
# 3. Initialize Weights
# -----------------------------

np.random.seed(42)
input_size = 30
hidden_size = 12
output_size = 3

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.1
epochs = 1000

losses = []
accuracies = []

# -----------------------------
# 4. Training Loop
# -----------------------------

for epoch in range(epochs):

    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(output_input)

    # Loss (Mean Squared Error)
    loss = np.mean((Y - output) ** 2)
    losses.append(loss)

    # Accuracy
    predictions = np.argmax(output, axis=1)
    true_labels = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions == true_labels)
    accuracies.append(accuracy)

    # Backpropagation
    output_error = Y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += np.dot(hidden_output.T, output_delta) * learning_rate
    b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    W1 += np.dot(X.T, hidden_delta) * learning_rate
    b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {accuracy:.2f}")

# -----------------------------
# 5. Plot Loss and Accuracy
# -----------------------------

plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(accuracies)
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# -----------------------------
# 6. Test the Model
# -----------------------------

labels = ['A', 'B', 'C']

test_index = 0  # Change to 1 or 2 for B or C
test_input = X[test_index].reshape(1, -1)

hidden = sigmoid(np.dot(test_input, W1) + b1)
output = sigmoid(np.dot(hidden, W2) + b2)

predicted_class = labels[np.argmax(output)]

print("\nPredicted Letter:", predicted_class)

# Display the image
plt.imshow(test_input.reshape(6, 5), cmap='gray')
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()

